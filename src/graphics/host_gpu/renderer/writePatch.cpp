// CPU stores into GPU-written buffer pages, without waiting for the GPU.
//
// A GPU-written page is protected, and a CPU store into it used to download the page and wait
// for the GPU before letting the store through (GT7 does this thousands of times a second).
// For a plain store of a register or constant, the GPU's data is not needed: the store is
// performed here instead. On the GPU thread (no GPU wait) the stored bytes go into guest memory
// through the backing view and into the GPU buffer, and the thread continues after the
// instruction. The page stays protected throughout, so no other thread can read it stale, and
// the GPU's other bytes on it stay where they are. Every other store (read-modify-write,
// string, masked, page-crossing, wide AVX, bursts on one page) keeps the readback path.
// KYTY_NO_WRITE_PATCH turns this off.

#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "kernel/memory.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <unordered_map>

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

bool WritePatchEnabled() {
	static const bool enabled = std::getenv("KYTY_NO_WRITE_PATCH") == nullptr;
	return enabled;
}

bool Gpr64(const CONTEXT& context, ZydisRegister reg, uint64_t& value) {
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

// Instructions whose memory destination receives the low bytes of their register or constant
// source.
bool IsPlainMove(ZydisMnemonic mnemonic) {
	switch (mnemonic) {
		case ZYDIS_MNEMONIC_MOV:
		case ZYDIS_MNEMONIC_MOVNTI:
		case ZYDIS_MNEMONIC_MOVSS:
		case ZYDIS_MNEMONIC_MOVSD:
		case ZYDIS_MNEMONIC_MOVUPS:
		case ZYDIS_MNEMONIC_MOVUPD:
		case ZYDIS_MNEMONIC_MOVAPS:
		case ZYDIS_MNEMONIC_MOVAPD:
		case ZYDIS_MNEMONIC_MOVDQU:
		case ZYDIS_MNEMONIC_MOVDQA:
		case ZYDIS_MNEMONIC_MOVNTDQ:
		case ZYDIS_MNEMONIC_MOVNTPS:
		case ZYDIS_MNEMONIC_MOVNTPD:
		case ZYDIS_MNEMONIC_MOVQ:
		case ZYDIS_MNEMONIC_MOVD:
		case ZYDIS_MNEMONIC_MOVLPS:
		case ZYDIS_MNEMONIC_MOVLPD:
		case ZYDIS_MNEMONIC_VMOVSS:
		case ZYDIS_MNEMONIC_VMOVSD:
		case ZYDIS_MNEMONIC_VMOVUPS:
		case ZYDIS_MNEMONIC_VMOVUPD:
		case ZYDIS_MNEMONIC_VMOVAPS:
		case ZYDIS_MNEMONIC_VMOVAPD:
		case ZYDIS_MNEMONIC_VMOVDQU:
		case ZYDIS_MNEMONIC_VMOVDQA:
		case ZYDIS_MNEMONIC_VMOVNTDQ:
		case ZYDIS_MNEMONIC_VMOVNTPS:
		case ZYDIS_MNEMONIC_VMOVNTPD:
		case ZYDIS_MNEMONIC_VMOVQ:
		case ZYDIS_MNEMONIC_VMOVD:
		case ZYDIS_MNEMONIC_VMOVLPS:
		case ZYDIS_MNEMONIC_VMOVLPD: return true;
		default: return false;
	}
}

struct PlainStore {
	uint64_t                address = 0;
	uint32_t                size    = 0;
	uint32_t                length  = 0;
	std::array<uint8_t, 32> bytes {};
};

// The upper 128 bits of YMM registers live in the context's extended (XSAVE) state.
bool YmmUpper(CONTEXT& context, uint32_t index, uint8_t* out) {
	DWORD64 features = 0;
	if (!GetXStateFeaturesMask(&context, &features) || (features & XSTATE_MASK_AVX) == 0) {
		static std::atomic_bool logged {false};
		if (!logged.exchange(true)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "write patch: no AVX state in the exception context (flags 0x{:x}, features 0x{:x})\n",
			    static_cast<uint32_t>(context.ContextFlags), static_cast<uint64_t>(features)));
		}
		return false;
	}
	DWORD length = 0;
	const auto* upper = static_cast<const M128A*>(LocateXStateFeature(&context, XSTATE_AVX, &length));
	if (upper == nullptr || length < (index + 1u) * sizeof(M128A)) {
		return false;
	}
	std::memcpy(out, &upper[index], sizeof(M128A));
	return true;
}

// Decodes a plain store and computes the bytes it writes, or returns false for anything else.
bool DecodePlainStore(CONTEXT& context, PlainStore& store) {
	ZydisDecoder decoder {};
	if (!ZYAN_SUCCESS(ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64))) {
		return false;
	}
	ZydisDecodedInstruction instruction {};
	ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
	if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(context.Rip),
	                                         ZYDIS_MAX_INSTRUCTION_LENGTH, &instruction, operands)) ||
	    !IsPlainMove(instruction.mnemonic) || instruction.operand_count_visible != 2 ||
	    instruction.address_width != 64 ||
	    (instruction.attributes &
	     (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE |
	      ZYDIS_ATTRIB_HAS_LOCK)) != 0 ||
	    (instruction.avx.mask.reg != ZYDIS_REGISTER_NONE &&
	     instruction.avx.mask.reg != ZYDIS_REGISTER_K0)) {
		return false;
	}
	const auto& destination = operands[0];
	const auto& source      = operands[1];
	if (destination.type != ZYDIS_OPERAND_TYPE_MEMORY ||
	    destination.actions != ZYDIS_OPERAND_ACTION_WRITE ||
	    destination.mem.type != ZYDIS_MEMOP_TYPE_MEM ||
	    destination.mem.segment == ZYDIS_REGISTER_FS || destination.mem.segment == ZYDIS_REGISTER_GS ||
	    destination.size == 0 || destination.size % 8 != 0 || destination.size > 256) {
		return false;
	}
	const uint32_t size = destination.size / 8u;

	// The value: the low bytes of a general register, an XMM or YMM register or a constant.
	std::array<uint8_t, 32> value {};
	if (source.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
		const uint64_t immediate = source.imm.is_signed ? static_cast<uint64_t>(source.imm.value.s)
		                                                : source.imm.value.u;
		if (size > 8) return false;
		std::memcpy(value.data(), &immediate, sizeof(immediate));
	} else if (source.type == ZYDIS_OPERAND_TYPE_REGISTER) {
		const auto reg_class = ZydisRegisterGetClass(source.reg.value);
		if (reg_class == ZYDIS_REGCLASS_GPR8 || reg_class == ZYDIS_REGCLASS_GPR16 ||
		    reg_class == ZYDIS_REGCLASS_GPR32 || reg_class == ZYDIS_REGCLASS_GPR64) {
			uint64_t full = 0;
			if (size > 8 ||
			    !Gpr64(context, ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64,
			                                                     source.reg.value),
			           full)) {
				return false;
			}
			if (source.reg.value == ZYDIS_REGISTER_AH || source.reg.value == ZYDIS_REGISTER_BH ||
			    source.reg.value == ZYDIS_REGISTER_CH || source.reg.value == ZYDIS_REGISTER_DH) {
				full >>= 8u;
			}
			std::memcpy(value.data(), &full, sizeof(full));
		} else if (reg_class == ZYDIS_REGCLASS_XMM || reg_class == ZYDIS_REGCLASS_YMM) {
			const bool ymm   = reg_class == ZYDIS_REGCLASS_YMM;
			const auto index = static_cast<uint32_t>(
			    source.reg.value - (ymm ? ZYDIS_REGISTER_YMM0 : ZYDIS_REGISTER_XMM0));
			if (index >= 16 || size > (ymm ? 32u : 16u) ||
			    (context.ContextFlags & CONTEXT_FLOATING_POINT) != CONTEXT_FLOATING_POINT) {
				return false;
			}
			const auto& xmm = context.FltSave.XmmRegisters[index];
			std::memcpy(value.data(), &xmm.Low, sizeof(xmm.Low));
			std::memcpy(value.data() + 8, &xmm.High, sizeof(xmm.High));
			if (ymm && size > 16u && !YmmUpper(context, index, value.data() + 16)) {
				return false;
			}
		} else {
			return false;
		}
	} else {
		return false;
	}

	uint64_t base  = 0;
	uint64_t index = 0;
	if (destination.mem.base == ZYDIS_REGISTER_RIP) {
		base = context.Rip + instruction.length;
	} else if (!Gpr64(context, destination.mem.base, base)) {
		return false;
	}
	if (!Gpr64(context, destination.mem.index, index)) {
		return false;
	}
	store.address = base + index * destination.mem.scale + static_cast<uint64_t>(destination.mem.disp.value);
	store.size    = size;
	store.length  = instruction.length;
	std::memcpy(store.bytes.data(), value.data(), size);
	return true;
}

// Logs each store site that keeps the readback path (once per instruction address), to see
// what the patching misses.
void NoteFallback(const CONTEXT& context, const char* reason) {
	static std::mutex                   mutex;
	static std::unordered_map<uint64_t, uint32_t> seen;
	{
		std::scoped_lock lock(mutex);
		if (seen.size() >= 200 || !seen.try_emplace(context.Rip, 0).second) return;
	}
	ZydisDecoder            decoder {};
	ZydisDecodedInstruction instruction {};
	ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
	const char*             mnemonic = "?";
	if (ZYAN_SUCCESS(ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64)) &&
	    ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(context.Rip),
	                                        ZYDIS_MAX_INSTRUCTION_LENGTH, &instruction, operands))) {
		mnemonic = ZydisMnemonicGetString(instruction.mnemonic);
	}
	Log::WriteToConsoleAndLog(fmt::format("write patch fallback: rip=0x{:x} {} ({})\n", context.Rip,
	                                      mnemonic != nullptr ? mnemonic : "?", reason));
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
	return ++burst.count > 64u;
}

} // namespace

bool RenderContext::TryBeginWritePatch(void* native_context, uint64_t fault_vaddr) noexcept {
	if (!WritePatchEnabled() || native_context == nullptr || GuestGpu::IsGpuThread() ||
	    CommandScheduler::InDeferredOperation() || m_gpu == nullptr || m_gpu->IsStopping() ||
	    !IsMapped(fault_vaddr, 1)) {
		return false;
	}
	const auto page = fault_vaddr & ~(PatchPageSize - 1u);
	if (!m_buffer_cache.IsRegionGpuModified(page, PatchPageSize)) {
		return false;
	}
	auto&      context = *static_cast<CONTEXT*>(native_context);
	PlainStore store;
	if (!DecodePlainStore(context, store)) {
		NoteFallback(context, "not a plain store");
		return false;
	}
	if (fault_vaddr < store.address || fault_vaddr >= store.address + store.size ||
	    !IsMapped(store.address, store.size)) {
		NoteFallback(context, "outside the store or unmapped");
		return false;
	}
	if (BurstLimited(page)) {
		PerfAdd(PerfCounter::WritePatchBegin, 0);
		return false;
	}
	// Images over the stored bytes see the CPU write, as on the readback path.
	m_texture_cache.InvalidateMemory(store.address, store.size);
	// Guest memory now; the GPU buffer before its next use (see BufferCache::StoreCpuBytes).
	if (!m_buffer_cache.StoreCpuBytes(store.address, store.bytes.data(), store.size)) {
		return false;
	}
	context.Rip += store.length;
	PerfAdd(PerfCounter::WritePatch, 0);
	return true;
}

bool RenderContext::FinishWritePatch(void*) noexcept {
	return false;
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
