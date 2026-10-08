#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fmt/format.h>
#include <mutex>
#include <utility>

namespace Libs::Graphics {

namespace {

struct NotedShader {
	uint64_t hash  = 0;
	uint32_t stage = 0;
	uint32_t count = 0;
};

std::mutex              g_noted_mutex;
std::deque<NotedShader> g_noted;

} // namespace

void DebugNoteShader(uint64_t hash, uint32_t stage) {
	std::scoped_lock lock(g_noted_mutex);
	const auto found = std::ranges::find_if(
	    g_noted, [&](const NotedShader& n) { return n.hash == hash && n.stage == stage; });
	NotedShader entry {hash, stage, 0};
	if (found != g_noted.end()) {
		entry = *found;
		g_noted.erase(found);
	}
	entry.count++;
	g_noted.push_back(entry);
	while (g_noted.size() > 64) {
		g_noted.pop_front();
	}
}

void DebugReportNotedShaders(const char* reason) {
	std::scoped_lock lock(g_noted_mutex);
	static constexpr const char* Stages[] = {"?", "VS", "PS", "FS", "CS", "MS", "LS", "HS", "TES"};
	std::string text = fmt::format("--- {}: shaders since last completed submit (oldest first, "
	                               "sync mode={}) ---\n",
	                               reason, DebugSyncSubmitEnabled() ? "on" : "off");
	for (const auto& n: g_noted) {
		text += fmt::format("  stage={} hash=0x{:016x} uses={}\n",
		                    n.stage < std::size(Stages) ? Stages[n.stage] : "?", n.hash, n.count);
	}
	Log::WriteToConsoleAndLog(text);
}

void DebugClearNotedShaders() {
	std::scoped_lock lock(g_noted_mutex);
	g_noted.clear();
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
		DebugReportNotedShaders("getSemaphoreCounterValue failed");
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

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
		DebugReportNotedShaders("waitSemaphores failed");
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
