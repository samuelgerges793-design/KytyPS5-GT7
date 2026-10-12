#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <string>
#include <atomic>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	void Wait(uint64_t tick);

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
};

// Experimental diagnostics: the shaders looked up since the last completed sync-mode submit
// are printed when the device is lost. KYTY_SYNC_SUBMIT=1 waits for every submit.
void DebugNoteShader(uint64_t hash, uint32_t stage);
void DebugNoteDispatch(uint32_t x, uint32_t y, uint32_t z, bool indirect, uint64_t args);
void DebugSubmitNoted(uint64_t tick);
void DebugCompletedUpTo(uint64_t tick);
void DebugReportNotedShaders(const char* reason, uint64_t completed_tick);
void DebugClearNotedShaders();
bool DebugSyncSubmitEnabled();
// Shaders listed in KYTY_SERIALIZE_SHADERS (comma-separated hashes) wait for recorded GPU work
// before their setup reads guest memory, and their dispatches run on their own.
bool SerializeShader(uint64_t hash);

// Where frame time goes: totals since the last report, printed with the fps every 5 s.
enum class PerfCounter {
	GpuWait,
	ShaderReadSync,
	SerializedDispatch,
	CommandThreadCompile,
	Readback,
	PipelineCreate,
	GpuThreadIdle,
	GpuThreadBlocked,
	DrawsWaitingForPipeline,
	// Driver pipeline cache result per created pipeline (time = creation time).
	PipelineCacheHit,
	PipelineCacheMiss,
	// Guest CPU accesses to protected pages (time includes any GPU readback and wait).
	CpuReadFault,
	CpuWriteFault,
	// CPU writes to GPU-written buffer pages: synchronous readbacks, and ones skipped.
	WriteReadback,
	WriteReadbackSkipped,
	// Plain CPU stores into GPU-written pages let through without a readback (writePatch.cpp).
	WritePatch,
	WritePatchBegin,
	// Bytes of storage buffers bound writable (the report shows MB/s); every byte is marked
	// GPU-written until the CPU reads it back.
	WritableBindBytes,
	// Command buffers read from GPU-covered memory: the wait and copy, and stale bytes found.
	CommandMemorySync,
	CommandMemoryStale,
	// Writable-binding verification: bytes copied back (MB/s), and pages released per second.
	WriteVerifyBytes,
	WriteVerifyReleased,
	// KYTY_WRITE_TRACK: bytes bound writable but predicted clean (MB/s), pages the GPU wrote
	// after all (marked late), and of those, pages the CPU had changed too.
	TrackSkipBytes,
	TrackMispredict,
	TrackConflict,
	TrackMergedBytes,
	// CPU stores and read faults on GPU-owned pages, by what the last checked submission saw.
	GpuOwnedWritten,
	GpuOwnedClean,
	GpuOwnedUntracked,
	// Bytes marked GPU-written per second, by source: tracked bindings (before prediction),
	// tracker overflow, written bindings of untracked programs, and other (DMA copies/fills
	// plus untracked bindings).
	MarkTrackedBytes,
	MarkOverflowBytes,
	MarkUntrackedBindingBytes,
	MarkOtherBytes,
	// GPU-owned pages released because the GPU never wrote them (pages per second).
	TrackReleased,
	TrackDownloadReleased,
	// CPU-touched GPU-owned pages (unique per flush), by tracking history.
	CandidateNoHistory,
	CandidateCommandClean,
	CandidateCommandWritten,
	CandidateClean,
	CandidateWritten,
	// Stores to GPU-owned pages the patching refused (each then reads back), by reason, and
	// explicit invalidations of GPU-owned memory.
	PatchRefusedGpuThread,
	PatchRefusedExtract,
	PatchRefusedRmw,
	PatchRefusedOtherInsn,
	PatchRefusedOther,
	ReadbackExplicit,
	// YMM stores whose context lacked AVX state (upper halves assumed zero).
	YmmAssumedZero,
	// KYTY_WRITE_PATCH_VERIFY: single-stepped stores whose real bytes matched / differed.
	PatchVerifyMatch,
	PatchVerifyMismatch,
	// Readbacks: guest bytes covered per second (window around each access).
	ReadbackWindowBytes,
	// Game submissions waiting for room in the GPU queue, and waits that timed out.
	SubmitQueueWait,
	SubmitQueueTimeout,
	// All-zero command buffers waited for (yielding to other queues), and given up on.
	ZeroBufferWait,
	ZeroBufferGaveUp,
	// End-of-pipe labels written when the GPU completes (KYTY_DEFERRED_LABELS).
	DeferredLabel,
	// Readbacks by the newest GPU work writing the range: none in flight, submitted, or still
	// being recorded (only the last needs a submit; none need the whole GPU drained).
	ReadbackNoWriter,
	ReadbackWriterSubmitted,
	ReadbackWriterRecording,
	// Readbacks copied on the second queue (time is the copy wait).
	ReadbackSideQueue,
	// Draws followed by a render-pass break and barrier because their shaders write buffers.
	DrawWriteBarrier,
	// Readback prefetch: readbacks served from earlier copies, copies dropped as stale, and
	// page pieces prefetched.
	PrefetchHit,
	PrefetchStale,
	PrefetchPages,
	Count
};
void PerfAdd(PerfCounter counter, uint64_t nanoseconds);

// Experimental switches that can change while the game runs: the value starts from the
// environment variable of the same name (or the default) and follows NAME=value lines in
// _kyty_live.txt in the working directory, reread with the perf report (every 5 s).
class LiveSetting {
public:
	LiveSetting(const char* name, uint64_t default_value);
	[[nodiscard]] uint64_t Get() const noexcept { return m_value.load(std::memory_order_relaxed); }

	const char*           m_name;
	uint64_t              m_initial = 0; // From the environment or the default.
	std::atomic<uint64_t> m_value;
};
void ReloadLiveSettings();
// Asks Windows not to throttle the emulator when its window is minimized or in the background.
void DisablePowerThrottling();
// Diagnostics (KYTY_WATCH_RANGE=hexaddr,hexsize): GPU writes into that range are logged.
bool WatchedRange(uint64_t vaddr, uint64_t size);
void NoteWatchedWrite(const std::string& what);
// Diagnostics (KYTY_HW_WATCH=addr[,addr...], up to 4): CPU debug registers on every thread catch
// each write to those 8-byte guest locations; direct-backing writes are logged too.
void HwWatchStart();
// Pages the command processor read commands from in the last 10 s: deferred image downloads
// must not write stale pixels over them (GT7 reuses image memory for command buffers).
void NoteCommandPages(uint64_t vaddr, uint64_t size);
bool IsRecentCommandPage(uint64_t vaddr);
// Diagnostics (KYTY_GPU_PROFILE=1): GPU time per draw, dispatch and copy kind, from timestamps
// written after each one; reported with the perf line. Labels are kind << 56 | shader hash.
bool        GpuProfileEnabled();
void        GpuProfileBegin(const vk::Device& device, float period, vk::CommandBuffer command, uint64_t tick,
                            uint64_t completed);
void        GpuProfileMark(vk::CommandBuffer command, char kind, uint64_t hash);
void        GpuProfileEnd(vk::CommandBuffer command);
uint64_t    GpuProfileLastShader(uint32_t stage);
std::string GpuProfileTake(double seconds);
bool HwWatchOverlaps(uint64_t vaddr, uint64_t size) noexcept;
void HwWatchNoteBackingWrite(uint64_t vaddr, const void* data, uint64_t size) noexcept;
// The GPU queue whose submission the GPU thread is processing (watch logs name it).
void SetCurrentGpuQueue(int queue);
void PerfReport(double fps);
// Background compile threads yield to the game and GPU threads.
void LowerCurrentThreadPriority();
// Tallies a guest page fault by page; the perf report lists the busiest pages.
void NoteFaultPage(uint64_t vaddr, bool write, uint64_t ns);
// Tallies a writable storage buffer binding; the report lists the largest by bytes bound.
void NoteWritableBinding(uint64_t vaddr, uint64_t size);
class PerfScope {
public:
	explicit PerfScope(PerfCounter counter);
	~PerfScope();
	PerfScope(const PerfScope&)            = delete;
	PerfScope& operator=(const PerfScope&) = delete;

private:
	PerfCounter m_counter;
	uint64_t    m_start;
};
// Compute shaders listed in KYTY_DUMP_SHADERS log their user data and the table at s[0:1].
bool DumpShaderInputs(uint64_t hash);
// VK_NV_device_diagnostic_checkpoints markers, enabled with KYTY_CHECKPOINTS=1.
void  DebugSetCheckpointsEnabled(bool enabled);
bool  DebugCheckpointsEnabled();
void* DebugRecordCheckpoint(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                            uint32_t arg2, uint32_t arg3, uint64_t arg4, uint64_t tick);
void  DebugReportCheckpoints(vk::Queue queue);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
