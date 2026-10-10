#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <fmt/format.h>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

struct NotedShader {
	uint64_t hash       = 0;
	uint32_t stage      = 0;
	uint32_t count      = 0;
	uint64_t max_groups = 0;
	uint32_t last_x = 0, last_y = 0, last_z = 0;
	uint32_t indirect   = 0;
	uint64_t last_args  = 0;
};

std::mutex                                   g_noted_mutex;
std::vector<NotedShader>                     g_recording;
std::map<uint64_t, std::vector<NotedShader>> g_submitted;
size_t                                       g_last_noted = SIZE_MAX;

std::string FormatNoted(const std::vector<NotedShader>& list) {
	static constexpr const char* Stages[] = {"?", "VS", "PS", "FS", "CS", "MS", "LS", "HS", "TES"};
	std::string text;
	for (const auto& n: list) {
		text += fmt::format("  stage={} hash=0x{:016x} uses={}",
		                    n.stage < std::size(Stages) ? Stages[n.stage] : "?", n.hash, n.count);
		if (n.max_groups != 0 || n.indirect != 0) {
			text += fmt::format(" groups(last)={}x{}x{} max_groups={} indirect={} args=0x{:x}",
			                    n.last_x, n.last_y, n.last_z, n.max_groups, n.indirect, n.last_args);
		}
		text += "\n";
	}
	return text;
}

} // namespace

void DebugNoteShader(uint64_t hash, uint32_t stage) {
	std::scoped_lock lock(g_noted_mutex);
	const auto found = std::ranges::find_if(
	    g_recording, [&](const NotedShader& n) { return n.hash == hash && n.stage == stage; });
	if (found != g_recording.end()) {
		found->count++;
		g_last_noted = static_cast<size_t>(found - g_recording.begin());
	} else if (g_recording.size() < 256) {
		g_recording.push_back({.hash = hash, .stage = stage, .count = 1});
		g_last_noted = g_recording.size() - 1;
	} else {
		g_last_noted = SIZE_MAX;
	}
}

void DebugNoteDispatch(uint32_t x, uint32_t y, uint32_t z, bool indirect, uint64_t args) {
	std::scoped_lock lock(g_noted_mutex);
	if (g_last_noted >= g_recording.size()) return;
	auto& n = g_recording[g_last_noted];
	if (indirect) {
		n.indirect++;
		n.last_args = args;
	} else {
		n.last_x = x;
		n.last_y = y;
		n.last_z = z;
		n.max_groups = std::max<uint64_t>(n.max_groups, uint64_t(x) * y * z);
	}
}

void DebugSubmitNoted(uint64_t tick) {
	std::scoped_lock lock(g_noted_mutex);
	g_submitted[tick] = std::move(g_recording);
	g_recording.clear();
	g_last_noted = SIZE_MAX;
	while (g_submitted.size() > 256) {
		g_submitted.erase(g_submitted.begin());
	}
}

void DebugCompletedUpTo(uint64_t tick) {
	std::scoped_lock lock(g_noted_mutex);
	g_submitted.erase(g_submitted.begin(), g_submitted.upper_bound(tick));
}

void DebugReportNotedShaders(const char* reason, uint64_t completed_tick) {
	std::scoped_lock lock(g_noted_mutex);
	std::string text = fmt::format("--- {}: GPU completed tick {}, {} submits unfinished ---\n",
	                               reason, completed_tick, g_submitted.size());
	int shown = 0;
	for (const auto& [tick, list]: g_submitted) {
		if (tick <= completed_tick) continue;
		text += fmt::format(" submit tick {}{}:\n", tick,
		                    shown == 0 ? " (oldest unfinished, where the GPU stopped)" : "");
		text += FormatNoted(list);
		if (++shown == 3) break;
	}
	text += " still recording:\n" + FormatNoted(g_recording);
	Log::WriteToConsoleAndLog(text);
}

void DebugClearNotedShaders() {
	std::scoped_lock lock(g_noted_mutex);
	g_recording.clear();
	g_submitted.clear();
	g_last_noted = SIZE_MAX;
}

void DebugPrintHostBacktrace(const char* reason) {
#ifdef _WIN32
	void*        frames[64] {};
	const USHORT count = RtlCaptureStackBackTrace(0, 64, frames, nullptr);
	const auto   base  = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	std::string  text  = fmt::format("--- host backtrace ({}), image base 0x{:x} ---\n", reason, base);
	for (USHORT i = 0; i < count; i++) {
		const auto address = reinterpret_cast<uintptr_t>(frames[i]);
		text += fmt::format("  #{:02} rva=0x{:x}\n", i, address - base);
	}
	Log::WriteToConsoleAndLog(text);
#else
	(void)reason;
#endif
}

namespace {

std::vector<uint64_t> ParseHashList(const char* name) {
	std::vector<uint64_t> list;
	if (const char* env = std::getenv(name); env != nullptr) {
		const std::string text(env);
		size_t start = 0;
		while (start < text.size()) {
			const auto end = std::min(text.find(',', start), text.size());
			if (end > start) list.push_back(std::stoull(text.substr(start, end - start), nullptr, 16));
			start = end + 1;
		}
	}
	return list;
}

} // namespace

bool DumpShaderInputs(uint64_t hash) {
	static const std::vector<uint64_t> hashes = ParseHashList("KYTY_DUMP_SHADERS");
	return !hashes.empty() && std::ranges::find(hashes, hash) != hashes.end();
}

namespace {

struct PerfTotals {
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> count {0};
};
std::array<PerfTotals, static_cast<size_t>(PerfCounter::Count)> g_perf;

uint64_t PerfNow() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

void PerfAdd(PerfCounter counter, uint64_t nanoseconds) {
	auto& totals = g_perf[static_cast<size_t>(counter)];
	totals.ns.fetch_add(nanoseconds, std::memory_order_relaxed);
	totals.count.fetch_add(1, std::memory_order_relaxed);
}

PerfScope::PerfScope(PerfCounter counter): m_counter(counter), m_start(PerfNow()) {}

PerfScope::~PerfScope() {
	PerfAdd(m_counter, PerfNow() - m_start);
}

void PerfReport(double fps) {
	static uint64_t last = PerfNow();
	const auto      now  = PerfNow();
	if (now - last < 5'000'000'000ull) return;
	const double seconds = static_cast<double>(now - last) / 1e9;
	last                 = now;
	static constexpr const char* Names[] = {"gpu-wait", "shader-read-sync", "serialized-dispatch",
	                                        "compile", "readback"};
	std::string text = fmt::format("perf: {:.1f} fps |", fps);
	for (size_t i = 0; i < g_perf.size(); i++) {
		const auto ns    = g_perf[i].ns.exchange(0, std::memory_order_relaxed);
		const auto count = g_perf[i].count.exchange(0, std::memory_order_relaxed);
		text += fmt::format(" {} {:.0f} ms/s ({:.0f}/s) |", Names[i],
		                    static_cast<double>(ns) / 1e6 / seconds,
		                    static_cast<double>(count) / seconds);
	}
	Log::WriteToConsoleAndLog(text + "\n");
}

bool SerializeShader(uint64_t hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		if (const char* env = std::getenv("KYTY_SERIALIZE_SHADERS"); env != nullptr) {
			const std::string text(env);
			size_t start = 0;
			while (start < text.size()) {
				const auto end = std::min(text.find(',', start), text.size());
				if (end > start) list.push_back(std::stoull(text.substr(start, end - start), nullptr, 16));
				start = end + 1;
			}
		}
		return list;
	}();
	return std::ranges::find(hashes, hash) != hashes.end();
}

namespace {

struct CheckpointInfo {
	uint64_t seq = 0, submit_id = 0, arg4 = 0, shader = 0, tick = 0;
	uint32_t op = 0, arg0 = 0, arg1 = 0, arg2 = 0, arg3 = 0;
};

constexpr size_t                         CheckpointRing = 1u << 16u;
std::atomic_bool                         g_checkpoints_enabled = false;
std::atomic<uint64_t>                    g_checkpoint_seq      = 0;
std::array<CheckpointInfo, CheckpointRing> g_checkpoints {};

} // namespace

void DebugSetCheckpointsEnabled(bool enabled) {
	g_checkpoints_enabled = enabled;
}

bool DebugCheckpointsEnabled() {
	return g_checkpoints_enabled;
}

void* DebugRecordCheckpoint(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                            uint32_t arg2, uint32_t arg3, uint64_t arg4, uint64_t tick) {
	const auto seq = ++g_checkpoint_seq;
	uint64_t   shader = 0;
	{
		std::scoped_lock lock(g_noted_mutex);
		if (g_last_noted < g_recording.size()) shader = g_recording[g_last_noted].hash;
	}
	g_checkpoints[seq % CheckpointRing] = {seq, submit_id, arg4, shader, tick, op, arg0, arg1, arg2, arg3};
	return reinterpret_cast<void*>(static_cast<uintptr_t>(seq));
}

void DebugReportCheckpoints(vk::Queue queue) {
	if (!g_checkpoints_enabled || queue == nullptr) return;
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, data.data());
	static constexpr const char* Ops[] = {"DispatchDirect", "DrawIndex", "DrawIndexAuto", "EopWrite",
	                                      "EopInterrupt", "EopWriteBack", "EopFlip", "EopWriteBackFlip",
	                                      "EopOnlyFlip", "DispatchIndirect", "Unknown"};
	std::string text = fmt::format("--- GPU checkpoints at device loss ({} stages, last seq recorded {}) ---\n",
	                               count, g_checkpoint_seq.load());
	for (const auto& d: data) {
		const auto seq  = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d.pCheckpointMarker));
		const auto& i   = g_checkpoints[seq % CheckpointRing];
		const char* name = i.op < std::size(Ops) ? Ops[i.op] : "?";
		text += fmt::format("  stage=0x{:x} marker={} -> {} submit={} tick={} shader=0x{:016x} "
		                    "args={},{},{},{},0x{:x}\n",
		                    static_cast<uint32_t>(d.stage), seq, i.seq == seq ? name : "(overwritten)",
		                    i.submit_id, i.tick, i.shader, i.arg0, i.arg1, i.arg2, i.arg3, i.arg4);
	}
	Log::WriteToConsoleAndLog(text);
}

bool DebugSyncSubmitEnabled() {
	static const bool enabled = std::getenv("KYTY_SYNC_SUBMIT") != nullptr;
	return enabled;
}

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		DebugReportNotedShaders("getSemaphoreCounterValue failed", KnownGpuTick());
		DebugReportCheckpoints(m_graphics.queue);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	DebugCompletedUpTo(counter);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	PerfScope  perf(PerfCounter::GpuWait);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		DebugReportNotedShaders("waitSemaphores failed", KnownGpuTick());
		DebugReportCheckpoints(m_graphics.queue);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
