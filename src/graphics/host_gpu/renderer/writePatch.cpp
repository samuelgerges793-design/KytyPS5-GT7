// CPU stores into GPU-written buffer pages, without waiting for the GPU.
//
// A GPU-written page is protected, and a CPU store into it used to download the page and wait
// for the GPU before letting the store through (GT7 does this thousands of times a second). For
// a plain store, the GPU's data is not needed: the page is opened for the one instruction
// (single-step), the bytes it stored are queued for upload into the GPU buffer, and the page is
// protected again. The guest page and the GPU buffer then both hold the store, and the GPU's
// other bytes on the page stay where they are. Anything else (read-modify-write, string and
// masked stores, page-crossing stores, bursts on one page) takes the readback path.
// KYTY_NO_WRITE_PATCH turns this off.

#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max

#include <Zydis/Zydis.h>
#endif

namespace Libs::Graphics {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr uint64_t PatchPageSize = 4096;
constexpr DWORD    TrapFlag      = 0x100;

struct PendingPatch {
	bool                               active  = false;
	uint64_t                           page    = 0;
	uint64_t                           address = 0;
	uint64_t                           size    = 0;
	std::array<uint8_t, PatchPageSize> before {};
};

thread_local PendingPatch t_pending;

bool WritePatchEnabled() {
	static const bool enabled = std::getenv("KYTY_NO_WRITE_PATCH") == nullptr;
	return enabled;
}

bool RegisterValue(const CONTEXT& context, ZydisRegister reg, uint64_t& value) {
	switch (reg) {
		case ZYDIS_REGISTER_NONE: value = 0; return true;
		case ZYDIS_REGISTER_RAX: value = context.Rax; return true;
		case ZYDIS_REGISTER_RCX: value = context.Rcx; return true;
		case ZYDIS_REGISTER_RDX: value = context.Rdx; return true;
		case ZYDIS_REGISTER_RBX: value = context.Rbx; return true;
		case ZYDIS_REGISTER_RSP: value = context.Rsp; return true;
		case ZYDIS_REGISTER_RBP: value = context.Rbp; return true;
		case ZYDIS_REGISTER_RSI: value = context.Rsi; return true;
		case ZYDIS_REGISTER_RDI: value = context.Rdi; return true;
		case ZYDIS_REGISTER_R8: value = context.R8; return true;
		case ZYDIS_REGISTER_R9: value = context.R9; return true;
		case ZYDIS_REGISTER_R10: value = context.R10; return true;
		case ZYDIS_REGISTER_R11: value = context.R11; return true;
		case ZYDIS_REGISTER_R12: value = context.R12; return true;
		case ZYDIS_REGISTER_R13: value = context.R13; return true;
		case ZYDIS_REGISTER_R14: value = context.R14; return true;
		case ZYDIS_REGISTER_R15: value = context.R15; return true;
		default: return false;
	}
}

// The one memory range a plain store instruction writes, or false for anything else.
bool DecodePlainStore(const CONTEXT& context, uint64_t& address, uint64_t& size) {
	ZydisDecoder decoder {};
	if (!ZYAN_SUCCESS(ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64))) {
		return false;
	}
	ZydisDecodedInstruction instruction {};
	ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
	if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(context.Rip),
	                                         ZYDIS_MAX_INSTRUCTION_LENGTH, &instruction, operands))) {
		return false;
	}
	if ((instruction.attributes &
	     (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE)) != 0 ||
	    instruction.address_width != 64 ||
	    (instruction.avx.mask.reg != ZYDIS_REGISTER_NONE &&
	     instruction.avx.mask.reg != ZYDIS_REGISTER_K0)) {
		return false;
	}
	const ZydisDecodedOperand* store = nullptr;
	for (uint32_t i = 0; i < instruction.operand_count; i++) {
		const auto& operand = operands[i];
		if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY ||
		    (operand.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) == 0) {
			continue;
		}
		// Read-modify-write needs the GPU's value; conditional stores may write nothing.
		if (store != nullptr || (operand.actions & ZYDIS_OPERAND_ACTION_MASK_READ) != 0 ||
		    (operand.actions & ZYDIS_OPERAND_ACTION_CONDWRITE) != 0 ||
		    operand.mem.type != ZYDIS_MEMOP_TYPE_MEM || operand.mem.segment == ZYDIS_REGISTER_FS ||
		    operand.mem.segment == ZYDIS_REGISTER_GS || operand.size == 0 || operand.size % 8 != 0) {
			return false;
		}
		store = &operand;
	}
	if (store == nullptr) {
		return false;
	}
	uint64_t base  = 0;
	uint64_t index = 0;
	if (store->mem.base == ZYDIS_REGISTER_RIP) {
		base = context.Rip + instruction.length;
	} else if (!RegisterValue(context, store->mem.base, base)) {
		return false;
	}
	if (!RegisterValue(context, store->mem.index, index)) {
		return false;
	}
	address = base + index * store->mem.scale + static_cast<uint64_t>(store->mem.disp.value);
	size    = store->size / 8u;
	return true;
}

// Bursts of stores into one page are cheaper as one readback, which hands the page to the CPU.
bool BurstLimited(uint64_t page) {
	struct Burst {
		std::chrono::steady_clock::time_point start;
		uint32_t                              count = 0;
	};
	static std::mutex                          mutex;
	static std::unordered_map<uint64_t, Burst> bursts;
	const auto                                 now = std::chrono::steady_clock::now();
	std::scoped_lock                           lock(mutex);
	if (bursts.size() > 4096) {
		bursts.clear();
	}
	auto& burst = bursts[page];
	if (burst.count == 0 || now - burst.start > std::chrono::milliseconds(50)) {
		burst = {now, 0};
	}
	return ++burst.count > 32u;
}

} // namespace

bool RenderContext::TryBeginWritePatch(void* native_context, uint64_t fault_vaddr) noexcept {
	if (!WritePatchEnabled() || native_context == nullptr || t_pending.active ||
	    GuestGpu::IsGpuThread() || CommandScheduler::InDeferredOperation() || m_gpu == nullptr ||
	    m_gpu->IsStopping() || !IsMapped(fault_vaddr, 1)) {
		return false;
	}
	const auto page = fault_vaddr & ~(PatchPageSize - 1u);
	if (!m_buffer_cache.IsRegionGpuModified(page, PatchPageSize)) {
		return false;
	}
	auto&    context = *static_cast<CONTEXT*>(native_context);
	uint64_t address = 0;
	uint64_t size    = 0;
	if (!DecodePlainStore(context, address, size) || fault_vaddr < address ||
	    fault_vaddr >= address + size || (address & ~(PatchPageSize - 1u)) != page ||
	    ((address + size - 1u) & ~(PatchPageSize - 1u)) != page || BurstLimited(page)) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(page, t_pending.before.data(), PatchPageSize)) {
		return false;
	}
	// Images over the stored bytes see the CPU write, as on the readback path.
	m_texture_cache.InvalidateMemory(address, size);
	t_pending.active  = true;
	t_pending.page    = page;
	t_pending.address = address;
	t_pending.size    = size;
	m_page_manager.SetTemporaryWrite(page, true);
	context.EFlags |= TrapFlag;
	return true;
}

bool RenderContext::FinishWritePatch(void* native_context) noexcept {
	if (!t_pending.active || native_context == nullptr) {
		return false;
	}
	auto& context = *static_cast<CONTEXT*>(native_context);
	context.EFlags &= ~TrapFlag;
	t_pending.active = false;
	m_page_manager.SetTemporaryWrite(t_pending.page, false);

	// Upload the instruction's store, plus any bytes another thread changed while the page was
	// open (those stores did not fault). Runs of changes close together upload as one.
	std::array<uint8_t, PatchPageSize> after {};
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.emplace_back(t_pending.address, t_pending.size);
	if (Libs::LibKernel::Memory::TryReadBacking(t_pending.page, after.data(), PatchPageSize)) {
		uint64_t run_begin = 0;
		uint64_t run_end   = 0;
		for (uint64_t i = 0; i < PatchPageSize; i++) {
			if (after[i] == t_pending.before[i]) continue;
			if (run_end != 0 && i <= run_end + 32u) {
				run_end = i + 1u;
				continue;
			}
			if (run_end != 0) ranges.emplace_back(t_pending.page + run_begin, run_end - run_begin);
			run_begin = i;
			run_end   = i + 1u;
		}
		if (run_end != 0) ranges.emplace_back(t_pending.page + run_begin, run_end - run_begin);
	}
	m_gpu->SendCommand([this, ranges = std::move(ranges)] {
		for (const auto& [address, size]: ranges) {
			m_buffer_cache.ApplyCpuWrite(address, size);
		}
	});
	PerfAdd(PerfCounter::WritePatch, 0);
	return true;
}

#else

bool RenderContext::TryBeginWritePatch(void*, uint64_t) noexcept {
	return false;
}

bool RenderContext::FinishWritePatch(void*) noexcept {
	return false;
}

#endif

} // namespace Libs::Graphics
