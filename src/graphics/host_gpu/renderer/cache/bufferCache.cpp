#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <bit>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		NoteGpuWrite(address, chunk);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	m_scheduler.ProfileMark('?', 0);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());
	m_scheduler.ProfileMark('R', 0);

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t first_store = 0;
	{
		std::scoped_lock lock(m_cpu_store_mutex);
		first_store = m_recent_store_count;
	}
	auto publish = [this, mapped, offset, total_size, buffer_address, first_store,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		std::scoped_lock lock(m_cpu_store_mutex);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		// Queued CPU stores are newer than this download: write them again on top.
		for (const auto& store: m_cpu_stores) {
			for (const auto& copy: copies) {
				const auto begin = buffer_address + copy.srcOffset;
				if (store.address < begin + copy.size && begin < store.address + store.size) {
					Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
					break;
				}
			}
		}
		// So are stores emulated after the copy was recorded that a flush already took from
		// the queue (the GPU copy predates their upload): write them again, oldest first.
		if (m_recent_store_count - first_store > m_recent_stores.size()) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				Log::WriteToConsoleAndLog("buffer download: store history overflowed while in flight\n");
			}
		}
		const auto oldest = std::max(first_store, m_recent_store_count -
		                                              std::min<uint64_t>(m_recent_store_count, m_recent_stores.size()));
		for (uint64_t sequence = oldest; sequence < m_recent_store_count; sequence++) {
			const auto& store = m_recent_stores[sequence % m_recent_stores.size()];
			for (const auto& copy: copies) {
				const auto begin = buffer_address + copy.srcOffset;
				if (store.address < begin + copy.size && begin < store.address + store.size) {
					Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
					break;
				}
			}
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_write_tracker(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                      ShaderRecompiler::IR::WriteTrackerBits / 8u),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_scheduler.SetPreSubmitHook(this);
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	SetVulkanObjectNameF(m_graphics.device, m_write_tracker.Handle(), "Write Tracker");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
	m_scheduler.SetPreSubmitHook(nullptr);
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	if (ShaderRecompiler::IR::WriteTrackingEnabled()) {
		std::scoped_lock lock(m_snapshot_mutex);
		if (!m_snapshot_candidates.empty()) {
			for (uint64_t page = vaddr >> 12u; page <= (vaddr + size - 1u) >> 12u; page++) {
				if (!m_snapshot_candidates.contains(page) || m_snapshots.contains(page) ||
				    m_memory_tracker.IsRegionCpuModified(page << 12u, 0x1000)) {
					continue;
				}
				auto bytes = std::make_unique<std::array<uint8_t, 0x1000>>();
				if (Libs::LibKernel::Memory::TryReadBacking(page << 12u, bytes->data(), bytes->size())) {
					m_snapshots.emplace(page, std::move(bytes));
				}
			}
		}
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::NoteGpuOwnedAccess(uint64_t vaddr) {
	if (!ShaderRecompiler::IR::WriteTrackingEnabled()) {
		return;
	}
	const auto page = vaddr >> 12u;
	auto       counter = PerfCounter::GpuOwnedUntracked;
	{
		std::scoped_lock lock(m_snapshot_mutex);
		if (m_release_candidates.size() < 65536) m_release_candidates.insert(page);
		if (m_track_last_written.contains(page)) {
			counter = PerfCounter::GpuOwnedWritten;
		} else if (m_track_last_tracked.contains(page)) {
			counter = PerfCounter::GpuOwnedClean;
		}
	}
	PerfAdd(counter, 0);
}

void BufferCache::SetPredictedInFlight(uint64_t page, bool in_flight) {
	std::scoped_lock lock(m_snapshot_mutex);
	if (in_flight) {
		m_snapshot_candidates.insert(page);
	} else {
		m_snapshot_candidates.erase(page);
	}
}

bool BufferCache::TryReleaseUnwrittenPage(uint64_t page) {
	// A page marked GPU-written only by tracked bindings the GPU never wrote holds the same bytes
	// on both sides (CPU stores to it were applied to both), once every submission that marked
	// it has been checked.
	static const bool release = std::getenv("KYTY_WRITE_TRACK_NO_RELEASE") == nullptr;
	const auto history = m_page_history.find(page);
	if (!release || history == m_page_history.end() || history->second.ever_written) {
		return false;
	}
	const auto marked = m_page_mark_generation.find(page);
	if (marked == m_page_mark_generation.end() || marked->second > m_track_checked_generation) {
		return false;
	}
	const auto address = page << 12u;
	if (!m_gpu_modified_ranges.Intersects(address, 0x1000)) {
		return false;
	}
	m_gpu_modified_ranges.Subtract(address, 0x1000);
	m_memory_tracker.UnmarkRegionAsGpuModified(address, 0x1000);
	return true;
}

bool BufferCache::CanReleaseAfterDownload(uint64_t page) const {
	// Opt-in: a run with it ended in a garbage command packet (Oct 11, cause not confirmed).
	static const bool release = std::getenv("KYTY_WRITE_TRACK_DOWNLOAD_RELEASE") != nullptr;
	const auto history = m_page_history.find(page);
	if (!release || history == m_page_history.end() || history->second.downloads >= 4u ||
	    m_predicted_in_flight.contains(page)) {
		return false;
	}
	const auto marked = m_page_mark_generation.find(page);
	return marked != m_page_mark_generation.end() && marked->second <= m_track_checked_generation &&
	       m_gpu_modified_ranges.Intersects(page << 12u, 0x1000);
}

void BufferCache::ReleaseAfterDownload(std::vector<uint64_t>& pages) {
	// GPU-owned pages the CPU keeps touching, which no unchecked submission marked: download them
	// once (a wait) and hand them back to the CPU. Contiguous pages of one buffer go together.
	const auto now = std::chrono::steady_clock::now();
	if (now - m_last_release_download < std::chrono::milliseconds(50) ||
	    m_scheduler.Current().IsInvalid()) {
		return;
	}
	m_last_release_download = now;
	std::ranges::sort(pages);
	uint64_t released = 0;
	uint32_t runs     = 0;
	for (size_t i = 0; i < pages.size() && runs < 64u;) {
		const auto  begin = pages[i] << 12u;
		const auto* owner = m_page_table.Find(begin >> PageTable::kPageBits);
		if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(begin, 0x1000)) {
			i++;
			continue;
		}
		auto&  buffer = m_slot_buffers[*owner];
		size_t j      = i + 1u;
		while (j < pages.size() && pages[j] == pages[j - 1u] + 1u &&
		       buffer.IsInBounds(pages[j] << 12u, 0x1000)) {
			j++;
		}
		const auto size = uint64_t {j - i} << 12u;
		(void)DownloadBufferMemory<false>(buffer, begin, size);
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, size);
		for (size_t k = i; k < j; k++) {
			auto& history        = m_page_history[pages[k]];
			history.ever_written = false;
			history.downloads++;
		}
		released += j - i;
		runs++;
		i = j;
	}
	if (released != 0) PerfAdd(PerfCounter::TrackDownloadReleased, released);
}

void BufferCache::ProcessPendingMerges() {
	if (!ShaderRecompiler::IR::WriteTrackingEnabled()) {
		return;
	}
	std::vector<uint64_t> pages;
	std::vector<uint64_t> candidates;
	{
		std::scoped_lock lock(m_snapshot_mutex);
		candidates.assign(m_release_candidates.begin(), m_release_candidates.end());
		m_release_candidates.clear();
		if (!m_pending_merges.empty() && !m_scheduler.Current().IsInvalid()) {
			pages.swap(m_pending_merges);
		}
	}
	if (!candidates.empty()) {
		uint64_t              released = 0;
		std::vector<uint64_t> downloads;
		for (const auto page: candidates) {
			const auto history = m_page_history.find(page);
			PerfAdd(history == m_page_history.end()  ? PerfCounter::CandidateNoHistory
			        : history->second.command         ? (history->second.ever_written
			                                                 ? PerfCounter::CandidateCommandWritten
			                                                 : PerfCounter::CandidateCommandClean)
			        : history->second.ever_written    ? PerfCounter::CandidateWritten
			                                          : PerfCounter::CandidateClean,
			        0);
			if (TryReleaseUnwrittenPage(page)) {
				released++;
			} else if (CanReleaseAfterDownload(page)) {
				downloads.push_back(page);
			}
		}
		if (released != 0) PerfAdd(PerfCounter::TrackReleased, released);
		if (!downloads.empty()) ReleaseAfterDownload(downloads);
	}
	for (const auto page: pages) {
		const auto address = page << 12u;
		if (m_gpu_modified_ranges.Intersects(address, 0x1000)) {
			continue; // Marked since by other work.
		}
		const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
		if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(address, 0x1000)) {
			continue;
		}
		// Upload the CPU's changes (only those, when saved bytes exist), then mark the page
		// GPU-written: CPU accesses read back the merged bytes.
		(void)SynchronizeBuffer(m_slot_buffers[*owner], address, 0x1000, true, false);
		m_gpu_modified_ranges.Add(address, 0x1000);
		std::scoped_lock lock(m_snapshot_mutex);
		m_snapshots.erase(page);
	}
}


// KYTY_FAST_READBACK=1 (experimental, GT7 still corrupts data at boot): readbacks with no pending
// writer copy on a second queue instead of draining the main queue; 2 = verify against a drain.
static LiveSetting fast_verify("KYTY_FAST_READBACK", 0);

// A readback whose range no pending GPU work writes: the buffer already holds the final bytes,
// so copy them on the second queue and wait only for that copy, not for the main queue to drain.
bool BufferCache::DownloadOnReadbackQueue(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	auto& master = m_scheduler.GetMasterSemaphore();
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    total_size += Common::AlignUp(end - start, 64);
		    });
	    });
	if (copies.empty()) {
		m_rb_nothing = true; // Like the draining path: nothing copied, nothing unmarked.
		return true;
	}
	m_rb_nothing = false;
	// The copied ranges reach past the access (whole GPU-written ranges): wait for their own
	// newest writers, or take the draining path if one is still being recorded.
	uint64_t needed = 0;
	for (const auto& copy: copies) {
		needed = std::max(needed, LastInflightWriteTick(buffer_address + copy.srcOffset, copy.size));
	}
	if (needed >= m_scheduler.CurrentTick()) {
		return false;
	}
	if (needed != 0) master.Wait(needed);
	master.Refresh();
	const auto done = master.KnownGpuTick();
	uint64_t   first_store = 0;
	{
		std::scoped_lock lock(m_cpu_store_mutex);
		while (!m_store_flushes.empty() && m_store_flushes.front().first <= done) {
			m_completed_store_seq = m_store_flushes.front().second;
			m_store_flushes.pop_front();
		}
		first_store = m_completed_store_seq;
		if (m_recent_store_count - first_store > m_recent_stores.size() / 2) {
			return false; // Too many stores to replay: take the draining path.
		}
	}
	constexpr uint64_t RbSize = 16 * MiB;
	if (total_size > RbSize) {
		return false;
	}
	const auto& device = m_graphics.device;
	// KYTY_RB_QUEUE=1: copy on the transfer-only family (copy engines) instead of a second
	// graphics queue. Buffers are not handed over between families (fine on NVIDIA).
	static LiveSetting rb_queue("KYTY_RB_QUEUE", 0);
	const uint32_t     kind  = rb_queue.Get() != 0 && m_graphics.transfer_queue ? 1u : 0u;
	const auto         queue = kind == 1 ? m_graphics.transfer_queue : m_graphics.readback_queue;
	auto&              m_rb_pool_k = m_rb_pool[kind];
	if (!m_rb_pool_k) {
		vk::CommandPoolCreateInfo pool {};
		pool.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool.queueFamilyIndex = kind == 1 ? m_graphics.transfer_family : m_graphics.queue_family;
		m_rb_pool_k           = device.createCommandPool(pool).value;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_rb_pool_k;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		m_rb_command[kind]          = device.allocateCommandBuffers(allocate).value.front();
		m_rb_fence[kind]            = device.createFence({}).value;
		if (!m_rb_download) m_rb_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst, RbSize);
	}
	// The ranges are no longer GPU-written once copied.
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });

	(void)m_rb_command[kind].reset();
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)m_rb_command[kind].begin(begin);
	m_rb_command[kind].copyBuffer(buffer.Handle(), m_rb_download->Handle(), static_cast<uint32_t>(copies.size()),
	                        copies.data());
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	m_rb_command[kind].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
	                             &after, 0, nullptr, 0, nullptr);
	(void)m_rb_command[kind].end();

	// Waiting on the completed main-queue value makes its writes visible to the copy.
	const auto                       semaphore = master.Handle();
	const vk::PipelineStageFlags     wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo  timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &done;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = done != 0 ? 1u : 0u;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &m_rb_command[kind];
	const auto wait_start = std::chrono::steady_clock::now();
	EXIT_NOT_IMPLEMENTED(queue.submit(1, &submit, m_rb_fence[kind]) != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(device.waitForFences(1, &m_rb_fence[kind], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(device.resetFences(1, &m_rb_fence[kind]) != vk::Result::eSuccess);
	PerfAdd(PerfCounter::ReadbackSideQueue, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_start).count()));
	m_rb_download->Invalidate(0, total_size);
	// KYTY_FAST_READBACK=2: also copy on the main queue after it drains and compare.
	if (fast_verify.Get() == 2) {
		auto verify = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                       vk::BufferUsageFlagBits::eTransferDst, total_size);
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Handle();
		vk::MemoryBarrier       before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
		                       &before, 0, nullptr, 0, nullptr);
		native.copyBuffer(buffer.Handle(), verify->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
		vk::MemoryBarrier after_verify {};
		after_verify.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after_verify.dstAccessMask = vk::AccessFlagBits::eHostRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
		                       &after_verify, 0, nullptr, 0, nullptr);
		m_scheduler.Wait(m_scheduler.CurrentTick());
		verify->Invalidate(0, total_size);
		std::vector<uint8_t> fast_image(m_rb_download->Mapped().data(), m_rb_download->Mapped().data() + total_size);
		std::vector<uint8_t> slow_image(verify->Mapped().data(), verify->Mapped().data() + total_size);
		{
			// Both get the stores the publish replays: only GPU-side differences remain.
			std::scoped_lock lock(m_cpu_store_mutex);
			const auto apply = [&](const auto& store) {
				for (const auto& copy: copies) {
					const auto start = buffer_address + copy.srcOffset;
					const auto lo = std::max<uint64_t>(store.address, start);
					const auto hi = std::min<uint64_t>(store.address + store.size, start + copy.size);
					if (lo >= hi) continue;
					const auto at = copy.dstOffset + (lo - start);
					std::memcpy(fast_image.data() + at, store.bytes.data() + (lo - store.address), hi - lo);
					std::memcpy(slow_image.data() + at, store.bytes.data() + (lo - store.address), hi - lo);
				}
			};
			for (uint64_t sequence = first_store; sequence < m_recent_store_count; sequence++) {
				apply(m_recent_stores[sequence % m_recent_stores.size()]);
			}
			for (const auto& store: m_cpu_stores) apply(store);
		}
		const auto* fast_bytes = fast_image.data();
		const auto* slow_bytes = slow_image.data();
		for (const auto& copy: copies) {
			if (std::memcmp(fast_bytes + copy.dstOffset, slow_bytes + copy.dstOffset, copy.size) == 0) continue;
			uint64_t first = 0;
			while (first < copy.size && fast_bytes[copy.dstOffset + first] == slow_bytes[copy.dstOffset + first]) first++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 40) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "fast readback MISMATCH at 0x{:x} fast {:02x} slow {:02x} (range 0x{:x}+0x{:x}) writer-done={} current={} known={};{}\n",
				    buffer_address + copy.srcOffset + first, fast_bytes[copy.dstOffset + first],
				    slow_bytes[copy.dstOffset + first], buffer_address + copy.srcOffset, copy.size, done,
				    m_scheduler.CurrentTick(), master.KnownGpuTick(),
				    DescribePages(buffer_address + copy.srcOffset + first, 1)));
				const auto at = buffer_address + copy.srcOffset + first;
				std::string writes;
				const auto  n = std::min<uint64_t>(m_debug_write_count, m_debug_writes.size());
				for (uint64_t k = 0; k < n; k++) {
					const auto& w = m_debug_writes[(m_debug_write_count - 1 - k) % m_debug_writes.size()];
					if (w.begin <= at && at < w.end && writes.size() < 300) {
						writes += fmt::format(" {}@{}[0x{:x}+0x{:x}]", w.kind, w.tick, w.begin, w.end - w.begin);
					}
				}
				Log::WriteToConsoleAndLog(fmt::format("  writes covering it (newest first):{} addr-write-tick={} open={}\n", writes,
				                                      m_address_write_tick, m_address_write_open));
			}
			PerfAdd(PerfCounter::PatchVerifyMismatch, 0);
		}
		std::memcpy(m_rb_download->Mapped().data(), verify->Mapped().data(), total_size); // Publish the drained copy.
		m_scheduler.DeferOperation([owner = std::move(verify)]() mutable { owner.reset(); });
	}

	const auto*      mapped = m_rb_download->Mapped().data();
	std::scoped_lock lock(m_cpu_store_mutex);
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset, mapped + copy.dstOffset, copy.size);
	}
	const auto reapply = [&](const auto& store) {
		for (const auto& copy: copies) {
			const auto start = buffer_address + copy.srcOffset;
			if (store.address < start + copy.size && start < store.address + store.size) {
				Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
				break;
			}
		}
	};
	// Stores whose upload the GPU has not run yet are newer than the copy, oldest first.
	for (uint64_t sequence = first_store; sequence < m_recent_store_count; sequence++) {
		reapply(m_recent_stores[sequence % m_recent_stores.size()]);
	}
	for (const auto& store: m_cpu_stores) reapply(store);
	return true;
}

uint64_t BufferCache::m_scheduler_tick_for_debug() {
	return m_scheduler.CurrentTick();
}

uint32_t BufferCache::WorkInCurrentTick() {
	return m_work_tick == m_scheduler.CurrentTick() ? m_work_count : 0;
}

void BufferCache::CommitGpuWrites() {
	const auto tick = m_scheduler.CurrentTick();
	if (m_work_tick != tick) {
		m_work_tick  = tick;
		m_work_count = 0;
	}
	m_work_count++;
	for (size_t i = m_open_writes; i < m_inflight_writes.size(); i++) m_inflight_writes[i].tick = tick;
	m_open_writes = m_inflight_writes.size();
	m_address_write_open = false;
	if (m_address_write_tick != 0) m_address_write_tick = std::max(m_address_write_tick, tick);
}

void BufferCache::NoteAddressWrites() {
	LogWriteMark(0, UINT64_MAX);
	m_address_write_epoch++;
	m_address_write_tick = m_scheduler.CurrentTick();
	m_address_write_open = true;
}

void BufferCache::LogWriteMark(uint64_t begin, uint64_t end) {
	m_write_marks.push_back({begin, end, ++m_write_seq});
	if (m_write_marks.size() > 65536) m_write_marks.pop_front();
}

bool BufferCache::TryPublishPrefetched(uint64_t vaddr, uint64_t size) {
	if (m_prefetched.empty()) return false;
	const auto first_page = vaddr >> 12u;
	const auto last_page  = (vaddr + size - 1u) >> 12u;
	// Every page the GPU still owns must have a copy that no recorded GPU write has made stale.
	// Copies of pages handed over since are dropped: their store replay would be old data.
	std::vector<uint64_t> owned;
	for (uint64_t page = first_page; page <= last_page; page++) {
		if (!m_memory_tracker.IsRegionGpuModified(page << 12u, 0x1000)) {
			m_prefetched.erase(page);
			continue;
		}
		owned.push_back(page);
		const auto found = m_prefetched.find(page);
		if (found == m_prefetched.end()) return false;
		const auto& entry = *found->second;
		bool        stale = m_write_marks.empty() ? false : m_write_marks.front().seq > entry.write_seq + 1u;
		uint32_t    scanned = 0;
		for (auto it = m_write_marks.rbegin(); !stale && it != m_write_marks.rend() && it->seq > entry.write_seq; ++it) {
			stale = (it->begin < ((page + 1u) << 12u) && (page << 12u) < it->end) || ++scanned > 8192u;
		}
		{
			std::scoped_lock lock(m_cpu_store_mutex);
			stale |= m_recent_store_count - entry.first_store > m_recent_stores.size() / 2;
		}
		if (stale) {
			m_prefetched.erase(found);
			PerfAdd(PerfCounter::PrefetchStale, 0);
			return false;
		}
	}
	// KYTY_READBACK_PREFETCH_VERIFY=1: compare each copy with the GPU's current bytes (drains the
	// queue) and log differences with the GPU writes recorded on them.
	static LiveSetting prefetch_verify("KYTY_READBACK_PREFETCH_VERIFY", 0);
	if (prefetch_verify.Get() != 0) {
		auto& buffer = m_slot_buffers[FindBuffer(first_page << 12u, (last_page - first_page + 1u) << 12u)];
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total = 0;
		for (const auto page: owned) {
			for (const auto [offset, bytes]: m_prefetched[page]->ranges) {
				copies.emplace_back((page << 12u) + offset - buffer.CpuAddress(), total, bytes);
				total += Common::AlignUp(uint64_t {bytes}, 16);
			}
		}
		if (total != 0) {
			auto staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                        vk::BufferUsageFlagBits::eTransferDst, total);
			auto& command = m_scheduler.Current();
			command.EndRendering();
			const auto        native = command.Handle();
			vk::MemoryBarrier before {};
			before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
			before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
			                       &before, 0, nullptr, 0, nullptr);
			native.copyBuffer(buffer.Handle(), staging->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
			vk::MemoryBarrier after {};
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eHostRead;
			native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &after,
			                       0, nullptr, 0, nullptr);
			m_scheduler.Wait(m_scheduler.CurrentTick());
			staging->Invalidate(0, total);
			const auto* now = staging->Mapped().data();
			for (const auto& copy: copies) {
				const auto  address = buffer.CpuAddress() + copy.srcOffset;
				const auto& entry   = *m_prefetched[address >> 12u];
				const auto* old     = entry.bytes.data() + (address & 0xfffu);
				if (std::memcmp(old, now + copy.dstOffset, copy.size) == 0) continue;
				uint64_t first = 0;
				while (old[first] == now[copy.dstOffset + first]) first++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1) < 30) {
					std::string writes;
					const auto  n = std::min<uint64_t>(m_debug_write_count, m_debug_writes.size());
					for (uint64_t k = 0; k < n && writes.size() < 240; k++) {
						const auto& w = m_debug_writes[(m_debug_write_count - 1 - k) % m_debug_writes.size()];
						if (w.begin <= address + first && address + first < w.end) {
							writes += fmt::format(" {}@{}", w.kind, w.tick);
						}
					}
					Log::WriteToConsoleAndLog(fmt::format(
					    "prefetch MISMATCH at 0x{:x} old {:02x} now {:02x} (copy seq {}, now seq {}, tick {}); writes:{}\n",
					    address + first, old[first], now[copy.dstOffset + first], entry.write_seq, m_write_seq,
					    m_scheduler.CurrentTick(), writes));
				}
				PerfAdd(PerfCounter::PatchVerifyMismatch, 0);
				// Publish the GPU's current bytes instead.
				std::memcpy(m_prefetched[address >> 12u]->bytes.data() + (address & 0xfffu), now + copy.dstOffset, copy.size);
			}
			m_scheduler.DeferOperation([owner = std::move(staging)]() mutable { owner.reset(); });
		}
	}
	if (owned.empty()) return false;
	std::scoped_lock lock(m_cpu_store_mutex);
	for (const auto page: owned) {
		const auto  found   = m_prefetched.find(page);
		const auto& entry   = *found->second;
		const auto  address = page << 12u;
		for (const auto [offset, bytes]: entry.ranges) {
			// Only bytes the GPU still owns: others were handed over or rewritten since.
			m_gpu_modified_ranges.ForEachInRange(address + offset, bytes, [&](uint64_t start, uint64_t end) {
				Libs::LibKernel::Memory::WriteBacking(start, entry.bytes.data() + (start - address), end - start);
			});
			m_gpu_modified_ranges.Subtract(address + offset, bytes);
		}
		const auto replay = [&](const auto& store) {
			if (store.address < address + 0x1000u && address < store.address + store.size) {
				Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
			}
		};
		for (uint64_t sequence = entry.first_store; sequence < m_recent_store_count; sequence++) {
			replay(m_recent_stores[sequence % m_recent_stores.size()]);
		}
		for (const auto& store: m_cpu_stores) replay(store);
		m_memory_tracker.UnmarkRegionAsGpuModified(address, 0x1000);
		m_prefetched.erase(found);
	}
	return true;
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	DebugNoteWrite(vaddr, size, 'b');
	LogWriteMark(vaddr, vaddr + size);
	const auto tick = m_scheduler.CurrentTick();
	if (!m_inflight_writes.empty()) {
		const auto& last = m_inflight_writes.back();
		if (last.tick == tick && last.begin == vaddr && last.end == vaddr + size) return;
	}
	if (m_inflight_writes.size() >= 16384 && m_open_writes > 0) {
		m_inflight_dropped_tick = std::max(m_inflight_dropped_tick, m_inflight_writes.front().tick);
		m_inflight_writes.pop_front();
		m_open_writes--;
	}
	m_inflight_writes.push_back({vaddr, vaddr + size, tick});
}

uint64_t BufferCache::LastInflightWriteTick(uint64_t vaddr, uint64_t size) {
	m_scheduler.GetMasterSemaphore().Refresh();
	const auto done = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	while (m_open_writes > 0 && m_inflight_writes.front().tick <= done) {
		m_inflight_writes.pop_front();
		m_open_writes--;
	}
	const auto current = m_scheduler.CurrentTick();
	uint64_t   last    = m_inflight_dropped_tick > done ? m_inflight_dropped_tick : 0;
	if (m_address_write_open) return current;
	if (m_address_write_tick > done) last = std::max(last, m_address_write_tick);
	for (size_t i = 0; i < m_inflight_writes.size(); i++) {
		const auto& write = m_inflight_writes[i];
		if (write.begin < vaddr + size && vaddr < write.end) {
			last = std::max(last, i >= m_open_writes ? current : write.tick);
		}
	}
	return last;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		FlushCpuWrites();
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		// Experimental (KYTY_NO_WRITE_READBACK): a CPU write to a GPU-written page drops the
		// GPU's data on that page instead of downloading it and waiting for the GPU. Faster,
		// but GPU-written bytes the CPU does not overwrite are lost.
		static const bool skip_write_readback = std::getenv("KYTY_NO_WRITE_READBACK") != nullptr;
		if (is_write && skip_write_readback) {
			m_memory_tracker.ForEachDownloadRange<false>(
			    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
				    m_gpu_modified_ranges.Subtract(address, bytes);
			    });
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			PerfAdd(PerfCounter::WriteReadbackSkipped, 0);
			return;
		}
		std::optional<PerfScope> write_readback;
		if (is_write) write_readback.emplace(PerfCounter::WriteReadback);
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		// The CPU sweeps through these buffers: one wait downloads the GPU-written pages around
		// the access too (KYTY_READBACK_WINDOW_KB, e.g. 64; default 0 = just the access), so the
		// next accesses do not each need one. Pages it hands back stay clean for CPU-write
		// tracking. Off: in GT7 it copied stale GPU data over command buffers the game had just
		// written in the same window (hardware watchpoints, Oct 12), freezing or crashing races.
		static LiveSetting window_kb("KYTY_READBACK_WINDOW_KB", 0);
		const uint64_t     window = window_kb.Get() == 0 ? 0 : std::bit_floor(window_kb.Get()) * 1024ull;
		uint64_t begin = vaddr;
		uint64_t end   = vaddr + size;
		if (window != 0) {
			begin = std::max(buffer.CpuAddress(), vaddr & ~(window - 1u));
			end   = std::min(buffer.CpuAddress() + buffer.Size(), (vaddr + size + window - 1u) & ~(window - 1u));
			begin = std::min(begin, vaddr);
			end   = std::max(end, vaddr + size);
		}
		const auto writer = LastInflightWriteTick(begin, end - begin);
		if (writer == 0) {
			PerfAdd(PerfCounter::ReadbackNoWriter, 0);
		} else {
			PerfAdd(writer < m_scheduler.CurrentTick() ? PerfCounter::ReadbackWriterSubmitted
			                                           : PerfCounter::ReadbackWriterRecording, 0);
		}
		if (fast_verify.Get() != 0 && m_graphics.readback_queue && writer < m_scheduler.CurrentTick()) {
			if (writer != 0) m_scheduler.GetMasterSemaphore().Wait(writer);
			if (DownloadOnReadbackQueue(buffer, begin, end - begin)) {
				if (!m_rb_nothing) {
					m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
					PerfAdd(PerfCounter::ReadbackWindowBytes, end - begin);
				}
				if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
				return;
			}
		}
		static LiveSetting prefetch_kb("KYTY_READBACK_PREFETCH_KB", 0);
		std::unique_ptr<Buffer>     prefetch;
		std::vector<vk::BufferCopy> prefetch_copies;
		uint64_t                    prefetch_seq   = 0;
		uint64_t                    prefetch_store = 0;
		if (window == 0 && prefetch_kb.Get() != 0) {
			// KYTY_READBACK_PREFETCH_USE=0: copies are made but never handed over (diagnostics).
			static LiveSetting prefetch_use("KYTY_READBACK_PREFETCH_USE", 1);
			if (prefetch_use.Get() != 0 && TryPublishPrefetched(vaddr, size)) {
				PerfAdd(PerfCounter::PrefetchHit, 0);
				if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
				return;
			}
			const uint64_t pf         = std::bit_floor(prefetch_kb.Get()) * 1024ull;
			const auto     pf_begin   = std::max(buffer.CpuAddress(), vaddr & ~(pf - 1u));
			const auto     pf_end     = std::min(buffer.CpuAddress() + buffer.Size(), (vaddr + size + pf - 1u) & ~(pf - 1u));
			const auto     page_begin = vaddr & ~uint64_t {0xfff};
			const auto     page_end   = (vaddr + size + 0xfffu) & ~uint64_t {0xfff};
			// Pages a bound but not yet recorded draw writes are left out: the copy would come first.
			const auto open_write = [&](uint64_t page_address) {
				if (m_address_write_open) return true;
				for (size_t i = m_open_writes; i < m_inflight_writes.size(); i++) {
					const auto& w = m_inflight_writes[i];
					if (w.begin < page_address + 0x1000u && page_address < w.end) return true;
				}
				return false;
			};
			uint64_t total = 0;
			if (pf_begin < pf_end) {
				m_memory_tracker.ForEachDownloadRange<false>(
				    pf_begin, pf_end - pf_begin, [&](uint64_t address, uint64_t bytes) noexcept {
					    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
						    for (uint64_t cursor = start; cursor < end;) {
							    const auto page_address = cursor & ~uint64_t {0xfff};
							    const auto stop         = std::min(end, page_address + 0x1000u);
							    if ((page_address < page_begin || page_address >= page_end) && !open_write(page_address)) {
								    prefetch_copies.emplace_back(cursor - buffer.CpuAddress(), total, stop - cursor);
								    total += Common::AlignUp(stop - cursor, 16);
							    }
							    cursor = stop;
						    }
					    });
				    });
			}
			if (!prefetch_copies.empty() && total <= 4 * MiB) {
				prefetch = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
				                                    vk::BufferUsageFlagBits::eTransferDst, total);
				auto& command = m_scheduler.Current();
				command.EndRendering();
				const auto        native = command.Handle();
				vk::MemoryBarrier before {};
				before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
				before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
				                       &before, 0, nullptr, 0, nullptr);
				native.copyBuffer(buffer.Handle(), prefetch->Handle(), static_cast<uint32_t>(prefetch_copies.size()),
				                  prefetch_copies.data());
				vk::MemoryBarrier after {};
				after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				after.dstAccessMask = vk::AccessFlagBits::eHostRead;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
				                       &after, 0, nullptr, 0, nullptr);
				prefetch_seq = m_write_seq;
				// Stores since the last flush are not in the GPU copy (they upload at the next flush,
				// after the copy): replay from there when the copy is handed over.
				std::scoped_lock lock(m_cpu_store_mutex);
				prefetch_store = m_last_flush_store;
			} else {
				prefetch_copies.clear();
			}
		}
		const bool downloaded = DownloadBufferMemory<false>(buffer, begin, end - begin);
		for (uint64_t page = begin >> 12u; !m_prefetched.empty() && page <= (end - 1u) >> 12u; page++) {
			m_prefetched.erase(page); // Handed over by this download.
		}
		if (downloaded) {
			m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
			PerfAdd(PerfCounter::ReadbackWindowBytes, end - begin);
		}
		if (prefetch) {
			// The download waited for the whole tick, which holds the prefetch copies too.
			if (downloaded) {
				prefetch->Invalidate(0, prefetch->Size());
				const auto* mapped = prefetch->Mapped().data();
				if (m_prefetched.size() > 8192) m_prefetched.clear();
				for (const auto& copy: prefetch_copies) {
					const auto address = buffer.CpuAddress() + copy.srcOffset;
					const auto page    = address >> 12u;
					auto&      entry   = m_prefetched[page];
					if (!entry || entry->write_seq != prefetch_seq) {
						entry              = std::make_unique<PrefetchedPage>();
						entry->write_seq   = prefetch_seq;
						entry->first_store = prefetch_store;
					}
					const auto offset = address & 0xfffu;
					std::memcpy(entry->bytes.data() + offset, mapped + copy.dstOffset, copy.size);
					entry->ranges.emplace_back(static_cast<uint16_t>(offset), static_cast<uint16_t>(copy.size));
				}
				PerfAdd(PerfCounter::PrefetchPages, prefetch_copies.size());
			}
			m_scheduler.DeferOperation([owner = std::move(prefetch)]() mutable { owner.reset(); });
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	m_scheduler.ProfileMark('?', 0);
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	NoteGpuWrite(overlap.CpuAddress(), overlap.Size()); // The copy is a GPU write into the new buffer.
	m_scheduler.ProfileMark('M', 0);
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	{
		// A new buffer's contents are not what any saved page bytes describe.
		std::scoped_lock lock(m_snapshot_mutex);
		std::erase_if(m_snapshots, [&](const auto& entry) {
			const auto address = entry.first << 12u;
			return address >= overlap.begin && address < overlap.end;
		});
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	std::unique_lock snapshot_lock(m_snapshot_mutex, std::defer_lock);
	if (ShaderRecompiler::IR::WriteTrackingEnabled()) snapshot_lock.lock();
	const bool has_snapshots = snapshot_lock.owns_lock() && !m_snapshots.empty();
	if (snapshot_lock.owns_lock() && !has_snapshots) snapshot_lock.unlock();
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    const auto add = [&](uint64_t begin, uint64_t count) {
			    copies.emplace_back(total_size, buffer.Offset(begin), count);
			    total_size += count;
		    };
		    if (!has_snapshots) {
			    add(address, bytes);
			    return;
		    }
		    std::array<uint8_t, 0x1000> guest {};
		    for (uint64_t cursor = address; cursor < address + bytes;) {
			    const auto page_end = (cursor & ~uint64_t {0xfff}) + 0x1000;
			    const auto end      = std::min(page_end, address + bytes);
			    const auto saved    = m_snapshots.find(cursor >> 12u);
			    if (saved == m_snapshots.end() ||
			        !Libs::LibKernel::Memory::TryReadBacking(cursor & ~uint64_t {0xfff}, guest.data(),
			                                                 guest.size())) {
				    add(cursor, end - cursor);
				    cursor = end;
				    continue;
			    }
			    const auto& before = *saved->second;
			    for (uint64_t i = cursor & 0xfffu; i < ((end - 1u) & 0xfffu) + 1u;) {
				    if (guest[i] == before[i]) {
					    i++;
					    continue;
				    }
				    auto run_end = i + 1u;
				    // Runs separated by short equal gaps upload together.
				    for (uint64_t gap = 0; run_end < ((end - 1u) & 0xfffu) + 1u && gap < 16u; run_end++) {
					    gap = guest[run_end] == before[run_end] ? gap + 1u : 0u;
				    }
				    while (guest[run_end - 1u] == before[run_end - 1u]) run_end--;
				    add((cursor & ~uint64_t {0xfff}) + i, run_end - i);
				    PerfAdd(PerfCounter::TrackMergedBytes, run_end - i);
				    i = run_end;
			    }
			    m_snapshots.erase(saved);
			    cursor = end;
		    }
	    },
	    [&]() noexcept {
		    if (snapshot_lock.owns_lock()) snapshot_lock.unlock();
		    source = UploadCopies(buffer, copies, total_size);
	    });
	if (source) {
		RecordBufferUpload(buffer, source, copies);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

std::string BufferCache::DescribePages(uint64_t vaddr, uint64_t size) {
	std::string text;
	for (uint64_t page = vaddr >> 12u; page <= (vaddr + size - 1u) >> 12u; page++) {
		const auto address = page << 12u;
		const auto history = m_page_history.find(page);
		bool saved = false;
		{
			std::scoped_lock lock(m_snapshot_mutex);
			saved = m_snapshots.contains(page);
		}
		text += fmt::format(" [0x{:x} saved={} reg={} gpu={} cpu={} inflight={}", address, saved ? 1 : 0,
		                    IsRegionRegistered(address, 0x1000) ? 1 : 0,
		                    m_gpu_modified_ranges.Intersects(address, 0x1000) ? 1 : 0,
		                    m_memory_tracker.IsRegionCpuModified(address, 0x1000) ? 1 : 0,
		                    m_predicted_in_flight.contains(page) ? 1 : 0);
		if (history != m_page_history.end()) {
			text += fmt::format(" clean={} misses={} cmd={}", history->second.clean,
			                    history->second.misses, history->second.command ? 1 : 0);
		}
		text += "]";
	}
	return text;
}

uint64_t BufferCache::SyncGpuWrittenRange(uint64_t vaddr, uint64_t size, bool force) {
	NoteCommandPages(vaddr, size);
	static const bool enabled = std::getenv("KYTY_NO_CP_SYNC") == nullptr;
	if (!enabled || size == 0 || !GuestRange {vaddr, size}.Valid() ||
	    !IsRegionRegistered(vaddr, size) || m_scheduler.Current().IsInvalid()) {
		return 0;
	}
	// Pages synced after the last raw-address write work cannot hold newer GPU bytes, unless a
	// write-tracking prediction left them unmarked and the GPU may have written them since.
	bool stale     = false;
	bool predicted = false;
	for (uint64_t page = vaddr >> 12u; page <= (vaddr + size - 1u) >> 12u; page++) {
		const auto found = m_command_page_epochs.find(page);
		stale |= found == m_command_page_epochs.end() || found->second != m_address_write_epoch;
		if (ShaderRecompiler::IR::WriteTrackingEnabled()) {
			predicted |= m_predicted_in_flight.contains(page);
			m_page_history[page].command = true;
		}
	}
	stale |= predicted || force;
	if (!stale) {
		return 0;
	}
	if (m_command_page_epochs.size() > 65536) {
		m_command_page_epochs.clear();
	}
	for (uint64_t page = vaddr >> 12u; page <= (vaddr + size - 1u) >> 12u; page++) {
		m_command_page_epochs[page] = m_address_write_epoch;
	}
	PerfScope perf(PerfCounter::CommandMemorySync);
	FlushCpuWrites();
	uint64_t first_store = 0;
	{
		std::scoped_lock lock(m_cpu_store_mutex);
		first_store = m_recent_store_count;
	}
	auto&      buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	auto       staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                              vk::BufferUsageFlagBits::eTransferDst, size);
	auto&      command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 1, &before, 0, nullptr, 0, nullptr);
	const vk::BufferCopy copy {buffer.Offset(vaddr), 0, size};
	m_scheduler.ProfileMark('?', 0);
	native.copyBuffer(buffer.Handle(), staging->Handle(), 1, &copy);
	m_scheduler.ProfileMark('S', 0);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
	                       &after, 0, nullptr, 0, nullptr);
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	staging->Invalidate(0, size);

	const auto*          gpu = staging->Mapped().data();
	std::vector<uint8_t> guest(size);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, guest.data(), size)) {
		return 0;
	}
	uint64_t differing = 0;
	// Stores other threads emulated during the wait are in guest memory and queued, but not in
	// the GPU copy: hold the queue and write them again on top, as downloads do.
	std::scoped_lock store_lock(m_cpu_store_mutex);
	for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < vaddr + size; page += 0x1000) {
		const auto begin = std::max(page, vaddr);
		const auto end   = std::min(page + 0x1000, vaddr + size);
		const auto* a    = gpu + (begin - vaddr);
		const auto* b    = guest.data() + (begin - vaddr);
		if (std::memcmp(a, b, end - begin) == 0) continue;
		uint64_t count = 0;
		for (uint64_t i = 0; i < end - begin; i++) count += a[i] != b[i];
		differing += count;
		// The CPU's own newer bytes win on pages it changed after the GPU buffer was filled.
		if (!m_memory_tracker.IsRegionCpuModified(page, 0x1000)) {
			// A GPU copy that is all zero where the guest holds commands is a stale copy, not GPU
			// output: writing it back emptied GT7's compute command buffers and froze race loads
			// (KYTY_CP_SYNC_ZERO_GUARD=0 turns the guard off).
			static LiveSetting zero_guard("KYTY_CP_SYNC_ZERO_GUARD", 1);
			// Only when the guest bytes start like a PM4 packet (type 3, sane length): other bytes
			// there are stale data, not commands.
			uint32_t guest_header = 0;
			std::memcpy(&guest_header, b, std::min<uint64_t>(4, end - begin));
			const bool looks_like_commands = (guest_header >> 30u) == 3u && ((guest_header >> 16u) & 0x3fffu) < 0x400u;
			if (zero_guard.Get() != 0 && looks_like_commands &&
			    std::all_of(a, a + (end - begin), [](uint8_t v) { return v == 0; })) {
				static std::atomic<uint32_t> guard_logged {0};
				if (guard_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					uint32_t first = 0;
					std::memcpy(&first, b, std::min<uint64_t>(4, end - begin));
					Log::WriteToConsoleAndLog(fmt::format(
					    "CP sync: kept guest commands at 0x{:x}+0x{:x} (first 0x{:08x}) over an all-zero GPU copy;{}\n",
					    begin, end - begin, first, DescribePages(begin, end - begin)));
				}
				PerfAdd(PerfCounter::ZeroBufferGaveUp, 0);
				continue;
			}
			Libs::LibKernel::Memory::WriteBacking(begin, a, end - begin);
		}
	}
	if (differing != 0) {
		for (const auto& store: m_cpu_stores) {
			if (store.address < vaddr + size && vaddr < store.address + store.size) {
				Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
			}
		}
		const auto oldest = std::max(first_store, m_recent_store_count -
		                                              std::min<uint64_t>(m_recent_store_count, m_recent_stores.size()));
		for (uint64_t sequence = oldest; sequence < m_recent_store_count; sequence++) {
			const auto& store = m_recent_stores[sequence % m_recent_stores.size()];
			if (store.address < vaddr + size && vaddr < store.address + store.size) {
				Libs::LibKernel::Memory::WriteBacking(store.address, store.bytes.data(), store.size);
			}
		}
	}
	m_scheduler.DeferOperation([owner = std::move(staging)]() mutable { owner.reset(); });
	if (differing != 0) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "CP sync: commands at 0x{:x} ({} bytes) differ from the GPU copy in {} bytes{}\n",
			    vaddr, size, differing, predicted ? " (predicted-clean page)" : ""));
		}
		PerfAdd(PerfCounter::CommandMemoryStale, differing);
	}
	return differing;
}

uint32_t BufferCache::AllocateWriteTrack(uint64_t vaddr, uint64_t size) {
	using ShaderRecompiler::IR::WriteTrackerBits;
	using ShaderRecompiler::IR::WriteTrackUntracked;
	const auto [found, inserted] = m_track_bases.try_emplace({vaddr, size}, WriteTrackUntracked);
	if (!inserted) {
		return found->second;
	}
	const auto first_page = vaddr >> 12u;
	const auto pages      = ((vaddr + size - 1u) >> 12u) - first_page + 1u;
	// One spare bit before each range: the shader counts pages from where the descriptor view
	// starts, up to 255 bytes earlier and possibly on the previous page.
	constexpr uint32_t half_bits = WriteTrackerBits / 2u;
	if (pages + 1u > half_bits - m_track_cursor) {
		return WriteTrackUntracked;
	}
	if (!m_write_tracker_cleared) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		native.fillBuffer(m_write_tracker.Handle(), 0, m_write_tracker.Size(), 0);
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr,
		                       0, nullptr);
		m_write_tracker_cleared = true;
	}
	const auto base = m_track_half * half_bits + m_track_cursor + 1u;
	m_track_cursor += static_cast<uint32_t>(pages) + 1u;
	found->second = base;
	m_track_ranges.push_back({first_page, static_cast<uint32_t>(pages), base});
	return base;
}

static uint32_t RequiredCleanSubmissions(uint8_t misses) {
	constexpr std::array<uint32_t, 4> required {4u, 16u, 64u, 250u};
	return required[std::min<size_t>(misses, required.size() - 1u)];
}

bool BufferCache::PredictClean(uint64_t page) const {
	const auto found = m_page_history.find(page);
	return found != m_page_history.end() && !found->second.command &&
	       found->second.clean >= RequiredCleanSubmissions(found->second.misses);
}

void BufferCache::RecordWriteTracking() {
	// The previous submission's half is complete once this submission has run.
	auto ranges    = std::move(m_previous_track_ranges);
	auto predicted = std::move(m_previous_track_predicted);
	const auto used = m_previous_track_cursor;
	m_previous_track_ranges    = std::move(m_track_ranges);
	m_previous_track_predicted = std::move(m_track_predicted);
	m_previous_track_cursor    = m_track_cursor;
	m_track_ranges.clear();
	m_track_predicted.clear();
	m_track_bases.clear();
	m_track_cursor = 0;
	m_track_half ^= 1u;
	const auto generation = m_track_generation - 1u; // The previous submission's.
	m_track_generation++;
	if (ranges.empty()) {
		return;
	}
	// The half to read is the one the next submission allocates from again.
	const uint64_t start = uint64_t {m_track_half} * (ShaderRecompiler::IR::WriteTrackerBits / 8u / 2u);
	const uint64_t bytes = uint64_t {(used + 32u) / 32u} * 4u;
	auto staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                        vk::BufferUsageFlagBits::eTransferDst, bytes);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 1, &before, 0, nullptr, 0, nullptr);
	const vk::BufferCopy copy {start, 0, bytes};
	native.copyBuffer(m_write_tracker.Handle(), staging->Handle(), 1, &copy);
	native.fillBuffer(m_write_tracker.Handle(), start, bytes, 0);
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
	                       1, &to_host, 0, nullptr, 0, nullptr);
	vk::MemoryBarrier to_shaders {};
	to_shaders.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_shaders.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	                       {}, 1, &to_shaders, 0, nullptr, 0, nullptr);
	const auto first_bit = static_cast<uint32_t>(start * 8u);
	for (auto& range: ranges) {
		range.base -= first_bit;
	}
	m_scheduler.DeferOperation([this, bytes, generation, ranges = std::move(ranges),
	                            predicted = std::move(predicted),
	                            owner = std::move(staging)]() mutable {
		owner->Invalidate(0, bytes);
		ProcessWriteTracking(ranges, predicted,
		                     reinterpret_cast<const uint32_t*>(owner->Mapped().data()), generation);
		owner.reset();
	});
}

void BufferCache::ProcessWriteTracking(const std::vector<TrackedRange>& ranges,
                                       const std::vector<uint64_t>& predicted, const uint32_t* bits,
                                       uint64_t generation) {
	// Overlapping bindings: a page counts as written when any of its bits is set.
	std::unordered_map<uint64_t, bool> written;
	for (const auto& range: ranges) {
		for (uint32_t i = 0; i < range.pages; i++) {
			const auto bit = range.base + i;
			written[range.first_page + i] |= ((bits[bit >> 5u] >> (bit & 31u)) & 1u) != 0;
		}
	}
	for (const auto page: predicted) {
		const auto in_flight = m_predicted_in_flight.find(page);
		if (in_flight != m_predicted_in_flight.end() && --in_flight->second == 0) {
			m_predicted_in_flight.erase(in_flight);
			SetPredictedInFlight(page, false);
			// Saved bytes describe the GPU copy only while the page's prediction is checked;
			// a page the GPU wrote keeps them for its merge.
			const auto found = written.find(page);
			if (found == written.end() || !found->second) {
				std::scoped_lock lock(m_snapshot_mutex);
				m_snapshots.erase(page);
			}
		}
	}
	uint64_t mispredicted = 0;
	uint64_t conflicts    = 0;
	for (const auto page: predicted) {
		const auto found = written.find(page);
		if (found == written.end() || !found->second) {
			continue;
		}
		found->second = false; // Handled once.
		auto& history = m_page_history[page];
		if (history.misses < UINT8_MAX) history.misses++;
		history.clean = 0;
		const auto address = page << 12u;
		if (m_gpu_modified_ranges.Intersects(address, 0x1000)) {
			continue; // Marked since by other work.
		}
		if (!m_memory_tracker.IsRegionCpuModified(address, 0x1000)) {
			// The guest copy is the last upload: the GPU's bytes become the page's contents.
			m_gpu_modified_ranges.Add(address, 0x1000);
			m_memory_tracker.MarkRegionAsGpuModified(address, 0x1000);
			mispredicted++;
			continue;
		}
		std::scoped_lock lock(m_snapshot_mutex);
		if (!m_snapshots.contains(page)) {
			// Changed by the CPU without saved bytes: its whole page will win over the GPU's.
			conflicts++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "write tracker: GPU and CPU both wrote predicted-clean page 0x{:x} (no saved bytes)\n",
				    address));
			}
		}
		m_pending_merges.push_back(page);
		mispredicted++;
	}
	{
		std::unordered_set<uint64_t> last_written;
		std::unordered_set<uint64_t> last_tracked;
		last_tracked.reserve(written.size());
		for (const auto& [page, was_written]: written) {
			last_tracked.insert(page);
			if (was_written) last_written.insert(page);
		}
		for (const auto page: predicted) {
			if (m_gpu_modified_ranges.Intersects(page << 12u, 0x1000)) last_written.insert(page);
		}
		std::scoped_lock lock(m_snapshot_mutex);
		m_track_last_written.swap(last_written);
		m_track_last_tracked.swap(last_tracked);
	}
	static const bool release = std::getenv("KYTY_WRITE_TRACK_NO_RELEASE") == nullptr;
	uint64_t          released = 0;
	for (const auto& [page, was_written]: written) {
		auto& history = m_page_history[page];
		if (was_written) {
			history.clean        = 0;
			history.ever_written = true;
			continue;
		}
		if (history.clean < UINT8_MAX) {
			history.clean++;
		}
	}
	m_track_checked_generation = std::max(m_track_checked_generation, generation);
	for (const auto& [page, was_written]: written) {
		if (release && !was_written && TryReleaseUnwrittenPage(page)) released++;
	}
	if (released != 0) PerfAdd(PerfCounter::TrackReleased, released);
	if (mispredicted != 0) PerfAdd(PerfCounter::TrackMispredict, mispredicted);
	if (conflicts != 0) PerfAdd(PerfCounter::TrackConflict, conflicts);
}

void BufferCache::RecordWriteVerification() {
	RecordWriteTracking();
	if (m_tick_written_ranges.empty()) {
		return;
	}
	auto ranges = std::move(m_tick_written_ranges);
	m_tick_written_ranges.clear();
	// Whole 4 KiB pages (the tracker's granularity), merged.
	for (auto& [address, size]: ranges) {
		const auto end = Common::AlignUp(address + size, 0x1000);
		address        = Common::AlignDown(address, 0x1000);
		size           = end - address;
	}
	std::ranges::sort(ranges);
	std::vector<std::pair<uint64_t, uint64_t>> merged;
	for (const auto& [address, size]: ranges) {
		if (!merged.empty() && address <= merged.back().first + merged.back().second) {
			auto& last  = merged.back();
			last.second = std::max(last.first + last.second, address + size) - last.first;
		} else {
			merged.emplace_back(address, size);
		}
	}
	// Split by owning buffer; skip what no buffer covers any more.
	struct Chunk {
		BufferId id;
		uint64_t address;
		uint64_t size;
	};
	std::vector<Chunk> chunks;
	uint64_t           total = 0;
	for (const auto& [address, size]: merged) {
		for (uint64_t cursor = address; cursor < address + size;) {
			const auto* owner = m_page_table.Find(cursor >> PageTable::kPageBits);
			if (owner == nullptr || !*owner) {
				cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			const auto& buffer = m_slot_buffers[*owner];
			const auto  end    = std::min(address + size, buffer.CpuAddress() + buffer.Size());
			if (end <= cursor) {
				cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			chunks.push_back({*owner, cursor, end - cursor});
			total += end - cursor;
			cursor = end;
		}
	}
	// Very large frames stay conservative rather than copying hundreds of megabytes.
	if (chunks.empty() || total > (uint64_t {256} << 20u)) {
		return;
	}
	auto staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                        vk::BufferUsageFlagBits::eTransferDst, total);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 1, &before, 0, nullptr, 0, nullptr);
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	uint64_t                                   offset = 0;
	for (const auto& chunk: chunks) {
		const auto&          buffer = m_slot_buffers[chunk.id];
		const vk::BufferCopy copy {buffer.Offset(chunk.address), offset, chunk.size};
		native.copyBuffer(buffer.Handle(), staging->Handle(), 1, &copy);
		pages.emplace_back(chunk.address, offset);
		pages.emplace_back(chunk.size, 0);
		offset += chunk.size;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
	                       &after, 0, nullptr, 0, nullptr);
	PerfAdd(PerfCounter::WriteVerifyBytes, total);
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.DeferOperation([this, tick, total, pages = std::move(pages),
	                            owner = std::move(staging)]() mutable {
		owner->Invalidate(0, total);
		VerifyWrites(tick, pages, owner->Mapped().data());
		owner.reset();
	});
}

void BufferCache::VerifyWrites(uint64_t tick, const std::vector<std::pair<uint64_t, uint64_t>>& pages,
                               const uint8_t* gpu) {
	std::array<uint8_t, 0x1000> guest {};
	uint64_t                    released = 0;
	// Pairs: (address, staging offset), (size, 0).
	for (size_t i = 0; i + 1 < pages.size(); i += 2) {
		const auto address = pages[i].first;
		const auto offset  = pages[i].second;
		const auto size    = pages[i + 1].first;
		for (uint64_t page = address; page < address + size; page += 0x1000) {
			const auto* gpu_page = gpu + offset + (page - address);
			const bool  marked   = m_gpu_modified_ranges.Intersects(page, 0x1000);
			const auto mark = m_page_last_mark.find(page >> 12u);
			if (mark == m_page_last_mark.end() || mark->second != tick) {
				continue; // Marked again by later work, or already handled.
			}
			m_page_last_mark.erase(mark);
			if (!marked || !Libs::LibKernel::Memory::TryReadBacking(page, guest.data(), guest.size())) {
				continue;
			}
			if (std::memcmp(gpu_page, guest.data(), guest.size()) != 0) {
				continue; // Written by the GPU: stays GPU-owned.
			}
			m_gpu_modified_ranges.Subtract(page, 0x1000);
			m_memory_tracker.UnmarkRegionAsGpuModified(page, 0x1000);
			released++;
		}
	}
	if (released != 0) {
		PerfAdd(PerfCounter::WriteVerifyReleased, released);
	}
}

bool BufferCache::StoreCpuBytes(uint64_t vaddr, const uint8_t* data, uint32_t size) {
	if (size == 0 || size > 32 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock(m_cpu_store_mutex);
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	auto& store   = m_cpu_stores.emplace_back();
	store.address = vaddr;
	store.size    = size;
	std::memcpy(store.bytes.data(), data, size);
	auto& recent    = m_recent_stores[m_recent_store_count % m_recent_stores.size()];
	recent.address  = vaddr;
	recent.size     = size;
	recent.sequence = m_recent_store_count++;
	std::memcpy(recent.bytes.data(), data, size);
	return true;
}

std::string BufferCache::DescribeRecentStores(uint64_t vaddr, uint64_t size) {
	std::scoped_lock lock(m_cpu_store_mutex);
	uint32_t    hits = 0;
	uint32_t    kept = 0;
	std::string examples;
	const auto  available = std::min<uint64_t>(m_recent_store_count, m_recent_stores.size());
	for (uint64_t i = 0; i < available; i++) {
		const auto& recent = m_recent_stores[(m_recent_store_count - 1u - i) % m_recent_stores.size()];
		if (recent.address >= vaddr + size || vaddr >= recent.address + recent.size) continue;
		std::array<uint8_t, 32> now {};
		const bool same = Libs::LibKernel::Memory::TryReadBacking(recent.address, now.data(), recent.size) &&
		                  std::memcmp(now.data(), recent.bytes.data(), recent.size) == 0;
		hits++;
		kept += same ? 1u : 0u;
		if (hits <= 6) {
			examples += fmt::format(" #{}@0x{:x}+{}{}", m_recent_store_count - 1u - recent.sequence,
			                        recent.address, recent.size, same ? "" : "(lost)");
		}
	}
	return fmt::format("{} of the last {} emulated stores hit it, {} still in memory;{}", hits, available,
	                   kept, examples);
}

void BufferCache::FlushCpuWrites() {
	ProcessPendingMerges();
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	{
		std::scoped_lock lock(m_cpu_store_mutex);
		if (m_cpu_stores.empty()) {
			m_last_flush_store = m_recent_store_count; // Every store so far is uploaded.
			return;
		}
		if (m_scheduler.Current().IsInvalid()) {
			return; // Kept for the next flush.
		}
		ranges.reserve(m_cpu_stores.size());
		for (const auto& store: m_cpu_stores) ranges.emplace_back(store.address, store.size);
		m_cpu_stores.clear();
		m_store_flushes.emplace_back(m_scheduler.CurrentTick(), m_recent_store_count);
		m_last_flush_store = m_recent_store_count;
	}
	std::ranges::sort(ranges);
	// Merge touching and overlapping stores only: bytes between them are stale guest copies of
	// GPU data. Then one upload per buffer.
	std::vector<std::pair<uint64_t, uint64_t>> merged;
	for (const auto& [address, size]: ranges) {
		if (!merged.empty() && address <= merged.back().first + merged.back().second) {
			auto& last  = merged.back();
			last.second = std::max(last.first + last.second, address + size) - last.first;
		} else {
			merged.emplace_back(address, size);
		}
	}
	std::vector<std::pair<BufferId, std::pair<uint64_t, uint64_t>>> targets;
	for (const auto& [address, size]: merged) {
		if (IsRegionRegistered(address, size)) {
			targets.emplace_back(FindBuffer(address, size), std::pair {address, size});
		}
	}
	for (size_t first = 0; first < targets.size();) {
		const auto id   = targets[first].first;
		auto&      buffer = m_slot_buffers[id];
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total = 0;
		size_t                      next  = first;
		for (; next < targets.size() && targets[next].first == id; next++) {
			const auto [address, size] = targets[next].second;
			copies.emplace_back(total, buffer.Offset(address), size);
			total += size;
		}
		if (const auto source = UploadCopies(buffer, copies, total)) {
			RecordBufferUpload(buffer, source, copies);
		}
		first = next;
	}
}

void BufferCache::RecordBufferUpload(Buffer& buffer, vk::Buffer source,
                                     std::span<const vk::BufferCopy> copies) {
	for (const auto& copy: copies) DebugNoteWrite(buffer.CpuAddress() + copy.dstOffset, copy.size, 'u');
	{
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		m_scheduler.ProfileMark('?', 0);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		m_scheduler.ProfileMark('U', 0);
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
}

// Uploads read guest memory through its unprotected backing: a page another range keeps
// read-protected for GPU-dirty data must not fault while the tracker runs its upload callback.
// A buffer can extend past the game's mappings (stream growth reserves room ahead), so a range
// may cross an unmapped gap; the sparse read fills gaps with zeros instead of faulting on them.
static void ReadUploadSource(uint8_t* dst, uint64_t address, uint64_t size) {
	if (Libs::LibKernel::Memory::TryReadBacking(address, dst, size) ||
	    Libs::LibKernel::Memory::TryReadSparseBacking(address, dst, size)) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "BufferCache: upload reads guest memory directly at 0x{:x} size 0x{:x}\n", address, size));
	}
	std::memcpy(dst, reinterpret_cast<const void*>(address), size);
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			ReadUploadSource(mapped + copy.srcOffset, address, copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		ReadUploadSource(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, uint32_t* write_track_base) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	static const bool verify  = std::getenv("KYTY_WRITE_VERIFY") != nullptr;
	static const bool predict = std::getenv("KYTY_WRITE_TRACK_NO_PREDICT") == nullptr;
	uint32_t          track   = ShaderRecompiler::IR::WriteTrackUntracked;
	if (is_written && write_track_base != nullptr) {
		track             = AllocateWriteTrack(vaddr, size);
		*write_track_base = track;
	}
	if (track != ShaderRecompiler::IR::WriteTrackUntracked && predict) {
		// Pages predicted clean are synchronized like reads; the rest are marked GPU-written.
		uint64_t run_begin = vaddr;
		bool     run_mark  = true;
		bool     first     = true;
		const auto end     = vaddr + size;
		for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < end; page += 0x1000) {
			const bool mark  = !PredictClean(page >> 12u);
			if (mark) m_page_mark_generation[page >> 12u] = m_track_generation;
			if (!mark) {
				m_track_predicted.push_back(page >> 12u);
				if (m_predicted_in_flight[page >> 12u]++ == 0) SetPredictedInFlight(page >> 12u, true);
			}
			const auto begin = std::max(page, vaddr);
			if (first) {
				run_mark = mark;
				first    = false;
			} else if (mark != run_mark) {
				(void)SynchronizeBuffer(buffer, run_begin, begin - run_begin, run_mark, is_texel_buffer);
				if (run_mark) { m_gpu_modified_ranges.Add(run_begin, begin - run_begin); NoteGpuWrite(run_begin, begin - run_begin); }
				else PerfAdd(PerfCounter::TrackSkipBytes, begin - run_begin);
				run_begin = begin;
				run_mark  = mark;
			}
		}
		(void)SynchronizeBuffer(buffer, run_begin, end - run_begin, run_mark, is_texel_buffer);
		if (run_mark) { m_gpu_modified_ranges.Add(run_begin, end - run_begin); NoteGpuWrite(run_begin, end - run_begin); }
		else PerfAdd(PerfCounter::TrackSkipBytes, end - run_begin);
		PerfAdd(PerfCounter::MarkTrackedBytes, size);
	} else {
		(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
		if (is_written) {
			m_gpu_modified_ranges.Add(vaddr, size);
			NoteGpuWrite(vaddr, size);
			for (uint64_t page = vaddr >> 12u;
			     ShaderRecompiler::IR::WriteTrackingEnabled() && page <= (vaddr + size - 1u) >> 12u; page++) {
				if (track != ShaderRecompiler::IR::WriteTrackUntracked) {
					m_page_mark_generation[page] = m_track_generation;
				} else {
					m_page_history[page].ever_written = true;
				}
			}
			PerfAdd(write_track_base == nullptr ? PerfCounter::MarkOtherBytes
			        : track != ShaderRecompiler::IR::WriteTrackUntracked ? PerfCounter::MarkTrackedBytes
			                                                             : PerfCounter::MarkOverflowBytes,
			        size);
		}
	}
	if (is_written) {
		if (verify) {
			const auto tick = m_scheduler.CurrentTick();
			for (uint64_t page = vaddr >> 12u; page <= (vaddr + size - 1u) >> 12u; page++) {
				m_page_last_mark[page] = tick;
			}
			m_tick_written_ranges.emplace_back(vaddr, size);
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || !Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if (!is_gds && WatchedRange(vaddr, size)) {
		NoteWatchedWrite(fmt::format("fill 0x{:x}+0x{:x} = 0x{:x} ({})", vaddr, size, value,
		                             IsRegionGpuModified(vaddr, size) ? "GPU" : "CPU memset at record time"));
	}
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	CommitGpuWrites();
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	if (!dst_gds && WatchedRange(dst_vaddr, size)) {
		NoteWatchedWrite(fmt::format("copy 0x{:x} -> 0x{:x} +0x{:x}{}", src_vaddr, dst_vaddr, size,
		                             src_gds ? " (from GDS)" : ""));
	}
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	CommitGpuWrites();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	FlushCpuWrites();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
