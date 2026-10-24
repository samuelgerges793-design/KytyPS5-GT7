#include "graphics/host_gpu/renderer/cache/hostPageStats.h"

#include "common/logging/log.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::HostPageStats {

namespace {

constexpr uint32_t PageBits = 14;
constexpr uint64_t PageSize = uint64_t {1} << PageBits;
// Bindings this large are kept as ranges, not pages (a whole-heap descriptor covers thousands).
constexpr uint64_t GiantBinding = uint64_t {4} << 20u;
constexpr uint32_t NoFrame      = UINT32_MAX;

struct Page {
	uint32_t walk_frames        = 0; // frames of this window a walk uploaded bytes of it in
	uint32_t bind_frames        = 0; // ... a binding covered it in
	uint32_t bind_upload_frames = 0; // ... a binding's sync uploaded bytes of it in
	uint32_t write_frames       = 0; // ... the GPU wrote it through a binding in
	uint32_t last_walk          = NoFrame;
	uint32_t last_bind          = NoFrame;
	uint32_t last_bind_upload   = NoFrame;
	uint32_t last_write         = NoFrame;
	uint64_t walk_bytes         = 0;
};

struct State {
	std::mutex                         mutex;
	std::unordered_map<uint64_t, Page> pages;
	std::map<uint64_t, uint64_t>       giant; // giant bindings: begin -> end
	uint32_t                           frame        = 0;
	uint64_t                           walk_bytes   = 0;
	uint64_t                           walk_uploads = 0;
	uint64_t                           bindings     = 0;
	uint64_t                           gpu_writes   = 0;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
};

State& Stats() {
	static State state;
	return state;
}

template <typename F>
void ForPages(uint64_t vaddr, uint64_t size, F&& visit) {
	if (size == 0) {
		return;
	}
	for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1) >> PageBits; page++) {
		visit(page);
	}
}

void Mark(uint32_t frame, uint32_t& last, uint32_t& frames) {
	if (last != frame) {
		last = frame;
		frames++;
	}
}

std::string Megabytes(uint64_t bytes) {
	return std::to_string((bytes + (uint64_t {1} << 19u)) >> 20u) + " MB";
}

void Report(State& s) {
	const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - s.begin)
	                         .count();
	const uint32_t frames = std::max<uint32_t>(s.frame, 1u);
	const uint32_t hot_at = std::max<uint32_t>(2u, frames * 3u / 4u);
	struct Hot {
		uint64_t page;
		const Page* info;
	};
	std::vector<Hot> hot;
	uint64_t         walked        = 0;
	uint64_t         hot_bytes     = 0;
	uint64_t         hot_bound     = 0;
	uint64_t         hot_uploaded  = 0;
	uint64_t         hot_written   = 0;
	uint64_t         hot_in_giant  = 0;
	const auto       in_giant = [&](uint64_t page) {
        const auto address = page << PageBits;
        auto       it      = s.giant.upper_bound(address);
        if (it == s.giant.begin()) {
            return false;
        }
        --it;
        return it->second > address;
	};
	for (const auto& [page, info]: s.pages) {
		if (info.walk_frames == 0) {
			continue;
		}
		walked++;
		if (info.walk_frames < hot_at) {
			continue;
		}
		hot.push_back({page, &info});
		hot_bytes += info.walk_bytes;
		hot_bound += info.bind_frames != 0 ? 1u : 0u;
		hot_uploaded += info.bind_upload_frames != 0 ? 1u : 0u;
		hot_written += info.write_frames != 0 ? 1u : 0u;
		hot_in_giant += in_giant(page) ? 1u : 0u;
	}
	LOGF("Host pages (%.1f s, %u frames): the walks uploaded %s in %" PRIu64
	     " uploads (%s a frame) over %" PRIu64 " pages of 16 KB; hot (uploaded in %u frames or "
	     "more): %zu pages (%s of the game's memory), %.0f%% of the walks' bytes; of the hot, %" PRIu64
	     " under buffer bindings (%" PRIu64 " uploaded by them), %" PRIu64 " GPU-written through "
	     "them, %" PRIu64 " under bindings of 4 MB or more; %" PRIu64 " bindings, %" PRIu64
	     " GPU writes\n",
	     seconds, s.frame, Megabytes(s.walk_bytes).c_str(), s.walk_uploads,
	     Megabytes(s.walk_bytes / frames).c_str(), walked, hot_at, hot.size(),
	     Megabytes(hot.size() * PageSize).c_str(),
	     s.walk_bytes == 0 ? 0.0
	                       : 100.0 * static_cast<double>(hot_bytes) /
	                             static_cast<double>(s.walk_bytes),
	     hot_bound, hot_uploaded, hot_written, hot_in_giant, s.bindings, s.gpu_writes);
	// The largest runs of hot pages, by the bytes the walks uploaded in them.
	std::ranges::sort(hot, {}, &Hot::page);
	struct Run {
		uint64_t first = 0;
		uint64_t count = 0;
		uint64_t bytes = 0;
		bool     bound = false, uploaded = false, written = false, giant = false;
	};
	std::vector<Run> runs;
	for (const auto& h: hot) {
		if (runs.empty() || runs.back().first + runs.back().count != h.page) {
			runs.push_back({.first = h.page});
		}
		auto& run = runs.back();
		run.count++;
		run.bytes += h.info->walk_bytes;
		run.bound    = run.bound || h.info->bind_frames != 0;
		run.uploaded = run.uploaded || h.info->bind_upload_frames != 0;
		run.written  = run.written || h.info->write_frames != 0;
		run.giant    = run.giant || in_giant(h.page);
	}
	std::ranges::sort(runs, std::greater<> {}, &Run::bytes);
	std::string text;
	for (size_t i = 0; i < std::min<size_t>(runs.size(), 12); i++) {
		const auto& run = runs[i];
		text += fmt::format("{}0x{:x}+{} KB ({} a frame{}{}{}{})", i == 0 ? "" : ", ",
		                    run.first << PageBits, (run.count * PageSize) >> 10u,
		                    Megabytes(run.bytes / frames), run.bound ? ", bound" : "",
		                    run.uploaded ? ", uploaded by bindings" : "",
		                    run.written ? ", GPU-written" : "", run.giant ? ", in a giant binding" : "");
	}
	LOGF("Host pages: %zu runs of hot pages; the largest: %s\n", runs.size(), text.c_str());
	s.pages.clear();
	s.giant.clear();
	s.frame        = 0;
	s.walk_bytes   = 0;
	s.walk_uploads = 0;
	s.bindings     = 0;
	s.gpu_writes   = 0;
	s.begin        = std::chrono::steady_clock::now();
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_HOST_PAGE_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

void NoteWalkUpload(uint64_t vaddr, uint64_t size) {
	auto&            s = Stats();
	std::lock_guard  lock(s.mutex);
	s.walk_bytes += size;
	s.walk_uploads++;
	const auto end = vaddr + size;
	ForPages(vaddr, size, [&](uint64_t page) {
		auto& info = s.pages[page];
		Mark(s.frame, info.last_walk, info.walk_frames);
		info.walk_bytes += std::min(end, (page + 1) << PageBits) - std::max(vaddr, page << PageBits);
	});
}

void NoteBinding(uint64_t vaddr, uint64_t size, bool written, bool uploaded) {
	auto&           s = Stats();
	std::lock_guard lock(s.mutex);
	s.bindings++;
	if (size >= GiantBinding) {
		auto& end = s.giant[vaddr];
		end       = std::max(end, vaddr + size);
		return;
	}
	ForPages(vaddr, size, [&](uint64_t page) {
		// Pages a walk uploaded already, or that this binding uploads: the others are not hot.
		auto it = s.pages.find(page);
		if (it == s.pages.end()) {
			if (!uploaded) {
				return;
			}
			it = s.pages.emplace(page, Page {}).first;
		}
		auto& info = it->second;
		Mark(s.frame, info.last_bind, info.bind_frames);
		if (uploaded) {
			Mark(s.frame, info.last_bind_upload, info.bind_upload_frames);
		}
		if (written) {
			Mark(s.frame, info.last_write, info.write_frames);
		}
	});
}

void NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	auto&           s = Stats();
	std::lock_guard lock(s.mutex);
	s.gpu_writes++;
	if (size >= GiantBinding) {
		return;
	}
	ForPages(vaddr, size, [&](uint64_t page) {
		if (auto it = s.pages.find(page); it != s.pages.end()) {
			Mark(s.frame, it->second.last_write, it->second.write_frames);
		}
	});
}

void NoteFrame() {
	auto&           s = Stats();
	std::lock_guard lock(s.mutex);
	s.frame++;
	if (std::chrono::steady_clock::now() - s.begin >= std::chrono::seconds(10)) {
		Report(s);
	}
}

} // namespace Libs::Graphics::HostPageStats
