#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <chrono>
#include <string>
#include <memory>
#include <unordered_set>
#include <array>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <deque>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// write_track_base (written shader bindings only): receives the write-tracker bit of vaddr's
	// page, or IR::WriteTrackUntracked when the binding is marked GPU-written as a whole.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t  vaddr, uint64_t size,
	                                                        bool      is_written,
	                                                        bool      is_texel_buffer  = false,
	                                                        BufferId  id               = {},
	                                                        uint32_t* write_track_base = nullptr);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] Buffer* GetWriteTrackerBuffer() noexcept { return &m_write_tracker; }
	// Diagnostics: counts a CPU access to a GPU-owned page by what the last checked submission
	// saw: written by the GPU, tracked but clean, or not tracked.
	void NoteGpuOwnedAccess(uint64_t vaddr);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	// Tracking regions (base addresses, TRACKER_REGION_SIZE each) the CPU touched since the
	// last call; BDA preparation uploads the CPU's changes in their mapped parts.
	[[nodiscard]] std::vector<uint64_t> TakeCpuDirtyRegions() {
		return m_memory_tracker.TakeCpuDirtyRegions();
	}
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();
	// Any thread: performs a CPU store into a GPU-written buffer page through the backing view
	// and queues its upload; the GPU thread uploads the queue in one batch before the next
	// packet or readback (FlushCpuWrites). A readback that publishes over queued stores
	// writes them again, so a store is never lost to older GPU data.
	[[nodiscard]] bool StoreCpuBytes(uint64_t vaddr, const uint8_t* data, uint32_t size);
	void               FlushCpuWrites();
	// GPU thread: before the command processor reads [vaddr, vaddr + size) as commands. When a
	// GPU buffer covers it, waits for the GPU and copies the GPU's bytes into guest memory
	// (except on pages the CPU changed since), so commands the GPU wrote through raw addresses
	// are seen. Returns the number of bytes that differed.
	uint64_t           SyncGpuWrittenRange(uint64_t vaddr, uint64_t size, bool force = false);
	// Diagnostics: buffer-cache and write-tracking state of the pages of a range.
	[[nodiscard]] std::string DescribePages(uint64_t vaddr, uint64_t size);
	// Diagnostics: recent emulated CPU stores into a range, and whether memory still holds them.
	[[nodiscard]] std::string DescribeRecentStores(uint64_t vaddr, uint64_t size);
	// GPU thread: work recorded with a shader that writes through raw addresses. Command memory
	// synced since the last such work is skipped.
	void               NoteAddressWrites();
	// Called once the draw, dispatch or copy that bound written ranges is recorded.
	void               CommitGpuWrites();
	// Draws and dispatches recorded into the current tick so far.
	[[nodiscard]] uint32_t WorkInCurrentTick();
	// Writable bindings mark their whole range GPU-written. At the end of each command buffer,
	// the pages marked in it are copied back; when that work completes, pages whose GPU bytes
	// equal guest memory (the shaders did not change them) are released to the CPU, unless a
	// later command buffer marked them again. Opt-in (KYTY_WRITE_VERIFY): GT7 marks the same
	// pages again every frame, so it released little for its copying.
	void               RecordWriteVerification();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void RecordBufferUpload(Buffer& buffer, vk::Buffer source, std::span<const vk::BufferCopy> copies);
	struct PendingCpuStore {
		uint64_t                address = 0;
		uint32_t                size    = 0;
		std::array<uint8_t, 32> bytes {};
	};
	void                                   VerifyWrites(uint64_t tick,
	                                                    const std::vector<std::pair<uint64_t, uint64_t>>& pages,
	                                                    const uint8_t* gpu);
	std::vector<std::pair<uint64_t, uint64_t>> m_tick_written_ranges;
	std::unordered_map<uint64_t, uint64_t>     m_page_last_mark;
	// GPU write tracking (KYTY_WRITE_TRACK). Shaders set one bit per 4 KiB guest page they store
	// to. Submissions alternate between the two halves of the bitmap; a submission's half is
	// copied back and cleared at the end of the next one, so a draw recorded one submission after
	// its bindings were prepared is still counted. A page of writable
	// bindings that the GPU left unwritten enough submissions in a row (more after each wrong
	// guess) is no longer marked GPU-written at bind. When the GPU writes such a page after all,
	// it is marked GPU-written as soon as the submission completes, so CPU accesses read back.
	struct TrackedRange {
		uint64_t first_page = 0;
		uint32_t pages      = 0;
		uint32_t base       = 0;
	};
	struct PageHistory {
		uint8_t clean  = 0; // Submissions in a row without GPU writes.
		uint8_t misses = 0; // Times the GPU wrote the page after it was predicted clean.
		bool    command = false; // The command processor read commands from it: never predicted.
		// The GPU may have written it (a set bit, or marked by untracked work): its GPU copy can
		// differ from the guest's, so releasing it needs a download.
		bool    ever_written = false;
		uint8_t downloads    = 0; // Times released after a download.
	};
	[[nodiscard]] uint32_t AllocateWriteTrack(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool     PredictClean(uint64_t page) const;
	void                   RecordWriteTracking();
	void                   ProcessWriteTracking(const std::vector<TrackedRange>& ranges,
	                                            const std::vector<uint64_t>& predicted,
	                                            const uint32_t* bits, uint64_t generation);
	// Submissions are numbered by tracking generation; the last generation that marked each
	// page decides whether a checked generation can release it.
	uint64_t                                            m_track_generation = 1;
	uint64_t                                            m_track_checked_generation = 0;
	// GPU-owned pages the CPU touched (from fault threads), checked for release on the GPU thread.
	std::unordered_set<uint64_t>                        m_release_candidates;
	[[nodiscard]] bool TryReleaseUnwrittenPage(uint64_t page);
	[[nodiscard]] bool CanReleaseAfterDownload(uint64_t page) const;
	void               ReleaseAfterDownload(std::vector<uint64_t>& pages);
	std::chrono::steady_clock::time_point              m_last_release_download {};
	std::unordered_map<uint64_t, uint64_t>              m_page_mark_generation;
	Buffer                                              m_write_tracker;
	bool                                                m_write_tracker_cleared = false;
	uint32_t                                            m_track_cursor          = 0;
	uint32_t                                            m_track_half            = 0;
	std::map<std::pair<uint64_t, uint64_t>, uint32_t>   m_track_bases;
	std::vector<TrackedRange>                           m_track_ranges;
	std::vector<uint64_t>                               m_track_predicted;
	uint32_t                                            m_previous_track_cursor = 0;
	std::vector<TrackedRange>                           m_previous_track_ranges;
	std::vector<uint64_t>                               m_previous_track_predicted;
	std::unordered_map<uint64_t, PageHistory>           m_page_history;
	// Pages left unmarked by a prediction whose submission has not been checked yet (count of
	// predictions in flight).
	std::unordered_map<uint64_t, uint32_t>              m_predicted_in_flight;
	// Merging CPU and GPU writes to a predicted page: the CPU's first write fault on an in-flight
	// page saves its bytes (equal to the GPU buffer before GPU writes). Uploads of the page then
	// send only the bytes the CPU changed, so GPU-written bytes in the buffer survive; a page the
	// GPU wrote after all gets the CPU's changes uploaded and is then marked GPU-written.
	void                                                SetPredictedInFlight(uint64_t page, bool in_flight);
	void                                                ProcessPendingMerges();
	std::mutex                                          m_snapshot_mutex;
	std::unordered_set<uint64_t>                        m_snapshot_candidates;
	std::unordered_map<uint64_t, std::unique_ptr<std::array<uint8_t, 0x1000>>> m_snapshots;
	std::vector<uint64_t>                               m_pending_merges;
	std::unordered_set<uint64_t>                        m_track_last_written;
	std::unordered_set<uint64_t>                        m_track_last_tracked;
	uint64_t                               m_address_write_epoch = 1;
	// Tick of the last work that writes through raw addresses (anywhere in guest memory).
	uint64_t                               m_address_write_tick  = 0;
	bool                                   m_address_write_open  = false;
	std::unordered_map<uint64_t, uint64_t> m_command_page_epochs;
	std::mutex                   m_cpu_store_mutex;
	std::vector<PendingCpuStore> m_cpu_stores;
	struct RecentStore {
		uint64_t                address  = 0;
		uint32_t                size     = 0;
		uint64_t                sequence = 0;
		std::array<uint8_t, 32> bytes {};
	};
	std::array<RecentStore, 65536> m_recent_stores {};
	uint64_t                      m_recent_store_count = 0;
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// GPU writes recorded since the last completed tick (oldest first): which readbacks need to
	// wait for which work.
	struct InflightWrite {
		uint64_t begin;
		uint64_t end;
		uint64_t tick;
	};
	std::deque<InflightWrite> m_inflight_writes;
	uint64_t                  m_inflight_dropped_tick = 0;
	void                      NoteGpuWrite(uint64_t vaddr, uint64_t size);
	// Readback prefetch (KYTY_READBACK_PREFETCH_KB): GPU-written pages around a readback are
	// copied in the same wait but kept here, and handed to the guest only when it touches them,
	// if no GPU write was recorded on them since (m_write_marks).
	struct PrefetchedPage {
		std::array<uint8_t, 4096>                  bytes {};
		std::vector<std::pair<uint16_t, uint16_t>> ranges; // offset, size within the page
		uint64_t                                   write_seq   = 0;
		uint64_t                                   first_store = 0;
	};
	struct WriteMark {
		uint64_t begin;
		uint64_t end;
		uint64_t seq;
	};
	std::unordered_map<uint64_t, std::unique_ptr<PrefetchedPage>> m_prefetched;
	std::deque<WriteMark>     m_write_marks;
	uint64_t                  m_write_seq = 0;
	void                      LogWriteMark(uint64_t begin, uint64_t end);
	bool                      TryPublishPrefetched(uint64_t vaddr, uint64_t size);
	// Writes noted from m_open_writes on belong to work not recorded yet (a sync between binding
	// and the draw moves it to a later tick): they count as the current tick until committed.
	size_t                    m_open_writes = 0;
	uint64_t                  m_work_tick  = 0;
	uint32_t                  m_work_count = 0;
	// Diagnostics for KYTY_FAST_READBACK=2: every recorded GPU write into buffers.
	struct DebugWrite {
		uint64_t begin = 0, end = 0, tick = 0;
		char     kind  = '?';
	};
	std::array<DebugWrite, 8192> m_debug_writes {};
	uint64_t                     m_debug_write_count = 0;
	void DebugNoteWrite(uint64_t vaddr, uint64_t size, char kind) {
		auto& w = m_debug_writes[m_debug_write_count++ % m_debug_writes.size()];
		w       = {vaddr, vaddr + size, m_scheduler_tick_for_debug(), kind};
	}
	uint64_t                  m_scheduler_tick_for_debug();
	// Emulated stores flushed per tick (tick, store sequence at the flush): stores from
	// m_completed_store_seq on may not be in the GPU's copy yet.
	std::deque<std::pair<uint64_t, uint64_t>> m_store_flushes;
	uint64_t                  m_completed_store_seq = 0;
	uint64_t                  m_last_flush_store    = 0; // store sequence at the last FlushCpuWrites
	vk::CommandPool           m_rb_pool[2]    = {};
	vk::CommandBuffer         m_rb_command[2] = {};
	vk::Fence                 m_rb_fence[2]   = {};
	std::unique_ptr<Buffer>   m_rb_download;
	bool                      m_rb_nothing = false;
	bool                      DownloadOnReadbackQueue(Buffer& buffer, uint64_t vaddr, uint64_t size);
	uint64_t                  LastInflightWriteTick(uint64_t vaddr, uint64_t size);
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
