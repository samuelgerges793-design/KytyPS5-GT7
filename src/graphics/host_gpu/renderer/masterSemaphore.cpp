#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <fmt/format.h>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

struct NotedShader {
	uint64_t hash  = 0;
	uint32_t stage = 0;
	uint32_t count = 0;
};

std::mutex                                   g_noted_mutex;
std::vector<NotedShader>                     g_recording;
std::map<uint64_t, std::vector<NotedShader>> g_submitted;

std::string FormatNoted(const std::vector<NotedShader>& list) {
	static constexpr const char* Stages[] = {"?", "VS", "PS", "FS", "CS", "MS", "LS", "HS", "TES"};
	std::string text;
	for (const auto& n: list) {
		text += fmt::format("  stage={} hash=0x{:016x} uses={}\n",
		                    n.stage < std::size(Stages) ? Stages[n.stage] : "?", n.hash, n.count);
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
	} else if (g_recording.size() < 256) {
		g_recording.push_back({hash, stage, 1});
	}
}

void DebugSubmitNoted(uint64_t tick) {
	std::scoped_lock lock(g_noted_mutex);
	g_submitted[tick] = std::move(g_recording);
	g_recording.clear();
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

	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		DebugReportNotedShaders("waitSemaphores failed", KnownGpuTick());
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
