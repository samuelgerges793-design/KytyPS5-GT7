#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

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
	Count
};
void PerfAdd(PerfCounter counter, uint64_t nanoseconds);
void PerfReport(double fps);
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
