#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/memory.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <fstream>
#include <filesystem>
#include <windows.h>
#include <tlhelp32.h>
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
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>
#include <cstring>

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

namespace {
std::mutex&                       LiveMutex() { static std::mutex m; return m; }
std::vector<LiveSetting*>&        LiveRegistry() { static std::vector<LiveSetting*> r; return r; }
std::map<std::string, uint64_t>&  LiveFileValues() { static std::map<std::string, uint64_t> v; return v; }
} // namespace

LiveSetting::LiveSetting(const char* name, uint64_t default_value): m_name(name), m_value(default_value) {
	if (const char* env = std::getenv(name); env != nullptr) {
		m_value.store(std::strtoull(env, nullptr, 0), std::memory_order_relaxed);
	}
	m_initial = m_value.load(std::memory_order_relaxed);
	std::scoped_lock lock(LiveMutex());
	if (const auto found = LiveFileValues().find(name); found != LiveFileValues().end()) {
		m_value.store(found->second, std::memory_order_relaxed);
	}
	LiveRegistry().push_back(this);
}

void ReloadLiveSettings() {
	static std::filesystem::file_time_type last {};
	std::error_code                       error;
	const auto time = std::filesystem::last_write_time("_kyty_live.txt", error);
	if (error || time == last) {
		return;
	}
	last = time;
	std::ifstream                   file("_kyty_live.txt");
	std::map<std::string, uint64_t> values;
	for (std::string line; std::getline(file, line);) {
		const auto equals = line.find('=');
		if (equals == std::string::npos || line.empty() || line[0] == '#') continue;
		auto name = line.substr(0, equals);
		while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
		values[name] = std::strtoull(line.c_str() + equals + 1, nullptr, 0);
	}
	std::scoped_lock lock(LiveMutex());
	LiveFileValues() = values;
	for (auto* setting: LiveRegistry()) {
		// Names missing from the file return to their starting value.
		const auto found = values.find(setting->m_name);
		const auto value = found != values.end() ? found->second : setting->m_initial;
		if (value != setting->Get()) {
			setting->m_value.store(value, std::memory_order_relaxed);
			Log::WriteToConsoleAndLog(fmt::format("live: {}={}\n", setting->m_name, value));
		}
	}
	Log::WriteToConsoleAndLog("live: settings file read\n");
}

void DisablePowerThrottling() {
#ifdef _WIN32
	PROCESS_POWER_THROTTLING_STATE state {};
	state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
	state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
	state.StateMask   = 0;
	(void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
#endif
}

bool WatchedRange(uint64_t vaddr, uint64_t size) {
	static const std::pair<uint64_t, uint64_t> range = [] {
		std::pair<uint64_t, uint64_t> result {0, 0};
		if (const char* env = std::getenv("KYTY_WATCH_RANGE"); env != nullptr) {
			char* end     = nullptr;
			result.first  = std::strtoull(env, &end, 16);
			result.second = (end != nullptr && *end == ',') ? std::strtoull(end + 1, nullptr, 16) : 0x1000;
		}
		return result;
	}();
	return range.second != 0 && vaddr < range.first + range.second && range.first < vaddr + size;
}

static std::atomic<int> g_current_gpu_queue {-1};

void SetCurrentGpuQueue(int queue) {
	g_current_gpu_queue.store(queue, std::memory_order_relaxed);
}

void NoteWatchedWrite(const std::string& what) {
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 600) {
		Log::WriteToConsoleAndLog(fmt::format("watch [queue {}]: {}\n", g_current_gpu_queue.load(), what));
	}
}

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

struct WaitSite {
	uint64_t count = 0;
	uint64_t ns    = 0;
};
std::mutex                       g_wait_sites_mutex;
std::unordered_map<uintptr_t, WaitSite> g_wait_sites;

// The longest single GPU wait since the last report: near 2 s before a timeout means frames
// that are too slow, not a hang.
std::atomic<uint64_t> g_longest_wait_ns {0};

void NoteLongestWait(uint64_t ns) {
	auto current = g_longest_wait_ns.load(std::memory_order_relaxed);
	while (ns > current && !g_longest_wait_ns.compare_exchange_weak(current, ns, std::memory_order_relaxed)) {
	}
}

struct WritableBinding {
	uint64_t size = 0, count = 0, shader = 0;
};
std::mutex                                    g_writable_mutex;
std::unordered_map<uint64_t, WritableBinding> g_writable_bindings;

struct FaultPage {
	uint64_t reads = 0, writes = 0, ns = 0;
};
std::mutex                              g_fault_pages_mutex;
std::unordered_map<uint64_t, FaultPage> g_fault_pages;

void NoteWaitSite(uint64_t ns) {
#ifdef _WIN32
	void*      frames[4] {};
	const auto count = RtlCaptureStackBackTrace(2, 4, frames, nullptr);
	const auto base  = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	// The first frame outside the scheduler wrappers names who waited.
	uintptr_t site = 0;
	for (USHORT i = 0; i < count && site == 0; i++) {
		site = reinterpret_cast<uintptr_t>(frames[i]) - base;
	}
	std::scoped_lock lock(g_wait_sites_mutex);
	auto&            entry = g_wait_sites[site];
	entry.count++;
	entry.ns += ns;
#else
	(void)ns;
#endif
}


namespace {
struct HwWatchHit {
	std::atomic<uint32_t> ready {0};
	uint32_t              thread = 0;
	uint64_t              rip    = 0;
	uint64_t              dr6    = 0;
	uint64_t              values[4] {};
	uint64_t              time   = 0;
};
std::vector<uint64_t>          g_hw_watch;
std::array<HwWatchHit, 1024>   g_hw_hits;
std::atomic<uint64_t>          g_hw_hit_count {0};
uint64_t                       g_hw_hits_printed = 0;

uint64_t ReadGuest64(uint64_t address) {
	uint64_t value = 0;
	(void)Libs::LibKernel::Memory::TryReadBacking(address, &value, sizeof(value));
	return value;
}

LONG CALLBACK HwWatchHandler(EXCEPTION_POINTERS* info) {
	if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || (info->ContextRecord->Dr6 & 0xfu) == 0) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const auto index = g_hw_hit_count.fetch_add(1, std::memory_order_acq_rel);
	auto&      hit   = g_hw_hits[index % g_hw_hits.size()];
	hit.ready.store(0, std::memory_order_relaxed);
	hit.thread = GetCurrentThreadId();
	hit.rip    = info->ContextRecord->Rip;
	hit.dr6    = info->ContextRecord->Dr6;
	for (size_t i = 0; i < g_hw_watch.size() && i < 4; i++) {
		hit.values[i] = *reinterpret_cast<volatile uint64_t*>(g_hw_watch[i]);
	}
	hit.time = PerfNow();
	hit.ready.store(1, std::memory_order_release);
	info->ContextRecord->Dr6 = 0;
	return EXCEPTION_CONTINUE_EXECUTION;
}

void HwWatchApplyAll() {
	DWORD64 dr7 = 0;
	for (size_t i = 0; i < g_hw_watch.size() && i < 4; i++) {
		dr7 |= 1ull << (i * 2);              // local enable
		dr7 |= 0x9ull << (16 + i * 4);       // write only, 8 bytes
	}
	const auto self = GetCurrentThreadId();
	const auto pid  = GetCurrentProcessId();
	HANDLE     snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) return;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) continue;
		HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE,
		                           entry.th32ThreadID);
		if (thread == nullptr) continue;
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT context {};
			context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
			if (GetThreadContext(thread, &context) && context.Dr7 != dr7) {
				context.Dr0 = g_hw_watch.size() > 0 ? g_hw_watch[0] : 0;
				context.Dr1 = g_hw_watch.size() > 1 ? g_hw_watch[1] : 0;
				context.Dr2 = g_hw_watch.size() > 2 ? g_hw_watch[2] : 0;
				context.Dr3 = g_hw_watch.size() > 3 ? g_hw_watch[3] : 0;
				context.Dr7 = dr7;
				SetThreadContext(thread, &context);
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
	}
	CloseHandle(snap);
}

void HwWatchPrint() {
	const auto count = g_hw_hit_count.load(std::memory_order_acquire);
	if (count > g_hw_hits_printed + g_hw_hits.size()) {
		Log::WriteToConsoleAndLog(fmt::format("hwwatch: {} hits lost\n", count - g_hw_hits_printed - g_hw_hits.size()));
		g_hw_hits_printed = count - g_hw_hits.size();
	}
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	for (; g_hw_hits_printed < count; g_hw_hits_printed++) {
		auto& hit = g_hw_hits[g_hw_hits_printed % g_hw_hits.size()];
		if (hit.ready.load(std::memory_order_acquire) == 0) break;
		std::string line = fmt::format("hwwatch: #{} t={:.3f} thread {} rip 0x{:x}{} dr6 0x{:x}", g_hw_hits_printed,
		                               static_cast<double>(hit.time) / 1e9, hit.thread, hit.rip,
		                               hit.rip >= base && hit.rip < base + 0x10000000 ? fmt::format(" (emu+0x{:x})", hit.rip - base) : "",
		                               hit.dr6 & 0xfu);
		for (size_t i = 0; i < g_hw_watch.size() && i < 4; i++) line += fmt::format(" [{:x}]={:016x}", g_hw_watch[i], hit.values[i]);
		Log::WriteToConsoleAndLog(line + "\n");
	}
}
} // namespace

bool HwWatchOverlaps(uint64_t vaddr, uint64_t size) noexcept {
	for (const auto address: g_hw_watch) {
		if (address < vaddr + size && vaddr < address + 8) return true;
	}
	return false;
}

void HwWatchNoteBackingWrite(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	for (const auto address: g_hw_watch) {
		if (address < vaddr + size && vaddr < address + 8) {
			uint64_t value = 0;
			const auto offset = address >= vaddr ? address - vaddr : 0;
			std::memcpy(&value, static_cast<const uint8_t*>(data) + offset, std::min<uint64_t>(8, size - offset));
			Log::WriteToConsoleAndLog(fmt::format("hwwatch: backing write [{:x}] <- {:016x} (range 0x{:x}+0x{:x}, was {:016x}) t={:.3f} gpu-thread={}\n",
			                                      address, value, vaddr, size, ReadGuest64(address),
			                                      static_cast<double>(PerfNow()) / 1e9, GuestGpu::IsGpuThread()));
		}
	}
}

void HwWatchStart() {
	const char* env = std::getenv("KYTY_HW_WATCH");
	if (env == nullptr || !g_hw_watch.empty()) return;
	std::string text(env);
	size_t      start = 0;
	while (start < text.size() && g_hw_watch.size() < 4) {
		const auto end = std::min(text.find(',', start), text.size());
		if (end > start) g_hw_watch.push_back(std::stoull(text.substr(start, end - start), nullptr, 16) & ~uint64_t {7});
		start = end + 1;
	}
	if (g_hw_watch.empty()) return;
	AddVectoredExceptionHandler(1, HwWatchHandler);
	std::thread([] {
		for (;;) {
			HwWatchApplyAll();
			HwWatchPrint();
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}).detach();
	Log::WriteToConsoleAndLog(fmt::format("hwwatch: watching {} addresses\n", g_hw_watch.size()));
}

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
	ReloadLiveSettings();
	const double seconds = static_cast<double>(now - last) / 1e9;
	last                 = now;
	static constexpr const char* Names[] = {"gpu-wait",      "shader-read-sync", "serialized-dispatch",
	                                        "compile",       "readback",         "pipeline-create",
	                                        "gpu-idle",      "gpu-blocked",      "draws-waiting",
	                                        "pipe-cache-hit", "pipe-cache-miss",
	                                        "cpu-read-fault", "cpu-write-fault",
	                                        "write-readback", "write-readback-skipped",
	                                        "write-patch", "write-patch-burst",
	                                        "writable-bind-MB", "cp-sync",
	                                        "cp-stale-bytes", "verify-copy-MB",
	                                        "verify-released", "track-skip-MB",
	                                        "track-mispredict-pages", "track-conflict-pages",
	                                        "track-merged-bytes", "gpuown-written",
	                                        "gpuown-clean", "gpuown-untracked",
	                                        "mark-tracked-MB", "mark-overflow-MB",
	                                        "mark-untracked-binding-MB", "mark-other-MB",
	                                        "track-released-pages", "track-download-released-pages",
	                                        "cand-nohistory", "cand-cmd-clean", "cand-cmd-written",
	                                        "cand-clean", "cand-written", "refused-gpu-thread",
	                                        "refused-extract", "refused-rmw", "refused-other-insn",
	                                        "refused-other", "readback-explicit", "ymm-assumed-zero",
	                                        "verify-match", "verify-MISMATCH",
	                                        "readback-window-MB", "submit-queue-wait",
	                                        "submit-queue-timeout", "zero-ib-wait", "zero-ib-gave-up",
	                                        "deferred-labels"};
	std::string text = fmt::format("perf: {:.1f} fps | longest-gpu-wait {:.0f} ms |", fps,
	                               static_cast<double>(g_longest_wait_ns.exchange(0)) / 1e6);
	for (size_t i = 0; i < g_perf.size(); i++) {
		const auto ns    = g_perf[i].ns.exchange(0, std::memory_order_relaxed);
		const auto count = g_perf[i].count.exchange(0, std::memory_order_relaxed);
		text += fmt::format(" {} {:.0f} ms/s ({:.0f}/s) |", Names[i],
		                    static_cast<double>(ns) / 1e6 / seconds,
		                    static_cast<double>(count) / seconds);
	}
	{
		std::scoped_lock lock(g_wait_sites_mutex);
		std::vector<std::pair<uintptr_t, WaitSite>> sites(g_wait_sites.begin(), g_wait_sites.end());
		g_wait_sites.clear();
		std::ranges::sort(sites, [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		if (!sites.empty()) text += " waits by caller:";
		for (size_t i = 0; i < sites.size() && i < 5; i++) {
			text += fmt::format(" rva=0x{:x} {:.0f}ms/{}", sites[i].first,
			                    static_cast<double>(sites[i].second.ns) / 1e6, sites[i].second.count);
		}
	}
	{
		std::scoped_lock lock(g_fault_pages_mutex);
		std::vector<std::pair<uint64_t, FaultPage>> pages(g_fault_pages.begin(), g_fault_pages.end());
		g_fault_pages.clear();
		std::ranges::sort(pages, [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		if (!pages.empty()) text += fmt::format(" | fault pages ({}):", pages.size());
		for (size_t i = 0; i < pages.size() && i < 5; i++) {
			text += fmt::format(" 0x{:x} r{}/w{} {:.0f}ms", pages[i].first, pages[i].second.reads,
			                    pages[i].second.writes, static_cast<double>(pages[i].second.ns) / 1e6);
		}
	}
	{
		std::scoped_lock lock(g_writable_mutex);
		std::vector<std::pair<uint64_t, WritableBinding>> bindings(g_writable_bindings.begin(),
		                                                           g_writable_bindings.end());
		g_writable_bindings.clear();
		std::ranges::sort(bindings, [](const auto& a, const auto& b) {
			return a.second.size * a.second.count > b.second.size * b.second.count;
		});
		if (!bindings.empty()) text += fmt::format(" | writable bindings ({}):", bindings.size());
		for (size_t i = 0; i < bindings.size() && i < 5; i++) {
			text += fmt::format(" 0x{:x}+{:.1f}MBx{} sh={:016x}", bindings[i].first,
			                    static_cast<double>(bindings[i].second.size) / 1048576.0,
			                    bindings[i].second.count, bindings[i].second.shader);
		}
	}
	Log::WriteToConsoleAndLog(text + "\n");
}

void NoteWritableBinding(uint64_t vaddr, uint64_t size) {
	PerfAdd(PerfCounter::WritableBindBytes, size);
	uint64_t shader = 0;
	{
		std::scoped_lock lock(g_noted_mutex);
		if (g_last_noted < g_recording.size()) shader = g_recording[g_last_noted].hash;
	}
	std::scoped_lock lock(g_writable_mutex);
	auto&            binding = g_writable_bindings[vaddr];
	binding.size             = std::max(binding.size, size);
	binding.count++;
	binding.shader = shader;
}

void NoteFaultPage(uint64_t vaddr, bool write, uint64_t ns) {
	PerfAdd(write ? PerfCounter::CpuWriteFault : PerfCounter::CpuReadFault, ns);
	std::scoped_lock lock(g_fault_pages_mutex);
	auto&            page = g_fault_pages[vaddr & ~uint64_t {0xfff}];
	(write ? page.writes : page.reads)++;
	page.ns += ns;
}

void LowerCurrentThreadPriority() {
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
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

	const auto wait_start = PerfNow();
	// Short GPU waits are most of the waits: polling the counter wakes up sooner than a kernel
	// wait (KYTY_GPU_SPIN_US, microseconds to poll before blocking; 0 = block at once).
	static LiveSetting spin_us("KYTY_GPU_SPIN_US", 0);
	if (const auto spin = spin_us.Get(); spin != 0) {
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin);
		do {
			YieldProcessor();
			Refresh();
			if (IsFree(tick)) {
				PerfAdd(PerfCounter::GpuWait, PerfNow() - wait_start);
				NoteWaitSite(PerfNow() - wait_start);
				NoteLongestWait(PerfNow() - wait_start);
				return;
			}
		} while (std::chrono::steady_clock::now() < deadline);
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	const auto result     = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	PerfAdd(PerfCounter::GpuWait, PerfNow() - wait_start);
	NoteWaitSite(PerfNow() - wait_start);
	NoteLongestWait(PerfNow() - wait_start);
	if (result != vk::Result::eSuccess) {
		DebugReportNotedShaders("waitSemaphores failed", KnownGpuTick());
		DebugReportCheckpoints(m_graphics.queue);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
