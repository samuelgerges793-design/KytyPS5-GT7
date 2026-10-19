#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderCacheFile.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "kytyShaderCacheVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

// Persistent caches belong to one code version, device and driver. The driver also checks its
// own header; the prefix ties the data to the code that made it.
std::string CacheSignature(std::string_view format, std::string_view version,
                           const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("{}:{}:{:08x}:{:08x}:{:08x}:{}\n", format, version, properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	// Development builds share one driver cache: entries match the exact shader and
	// state, so translation changes miss instead of reusing stale pipelines.
	return CacheSignature("KytyPC1", "gt7-dev", properties);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// Experimental: set per shader lookup; the first memory read of its materialization waits for
// the GPU work recorded so far, so values written by earlier dispatches are not read stale.
thread_local bool g_sync_before_shader_read = false;

// A setup read that copied GPU-written images back mixes old and new values; read it again.
bool MaterializeFresh(const ShaderRecompiler::IR::ResourcePlan& plan,
                      const ShaderRecompiler::IR::SrtRuntime& runtime,
                      ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                      ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	(void)Libs::LibKernel::Memory::TakeSetupImageReadback();
	bool ok = ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
	if (Libs::LibKernel::Memory::TakeSetupImageReadback()) {
		ok = ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
	}
	return ok;
}

thread_local bool g_skip_draw = false;

bool SpecializationSyncEnabled() {
	static const bool enabled = std::getenv("KYTY_SPEC_SYNC") != nullptr;
	return enabled;
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	if (g_sync_before_shader_read) {
		g_sync_before_shader_read = false;
		Libs::LibKernel::Memory::SyncGpuForShaderReads();
	}
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

// Why new shader variants and pipelines keep appearing: each compile and each pipeline creation
// records the reasons it was needed, logged per event (first few hundred) and summarized per
// shader every minute. KYTY_NO_VARIANT_LOG turns it off.
class VariantLog {
public:
	static VariantLog& Get() {
		static VariantLog log;
		return log;
	}

	bool Enabled() const { return m_enabled; }

	void NoteShader(uint64_t hash, ShaderType stage, const std::vector<std::string>& reasons) {
		if (!m_enabled) return;
		std::scoped_lock lock(m_mutex);
		auto& entry = m_shaders[hash];
		entry.stage = stage;
		entry.variants++;
		for (const auto& reason: reasons) entry.reasons[reason]++;
		if (m_shader_lines++ < 400) {
			std::string text = fmt::format("variant: {} 0x{:016x} #{}:", static_cast<int>(stage), hash,
			                               entry.variants);
			for (const auto& reason: reasons) text += " " + reason;
			Log::WriteToConsoleAndLog(text + "\n");
		}
		MaybeSummarize();
	}

	void NotePipeline(const std::vector<std::string>& reasons) {
		if (!m_enabled) return;
		std::scoped_lock lock(m_mutex);
		m_pipelines++;
		for (const auto& reason: reasons) m_pipeline_reasons[reason]++;
		if (m_pipeline_lines++ < 300) {
			std::string text = "pipeline variant:";
			for (const auto& reason: reasons) text += " " + reason;
			Log::WriteToConsoleAndLog(text + "\n");
		}
		MaybeSummarize();
	}

private:
	struct ShaderStats {
		ShaderType                      stage    = ShaderType::Unknown;
		uint32_t                        variants = 0;
		std::map<std::string, uint32_t> reasons;
	};

	static std::string TopReasons(const std::map<std::string, uint32_t>& reasons, size_t count) {
		std::vector<std::pair<std::string, uint32_t>> sorted(reasons.begin(), reasons.end());
		std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string text;
		for (size_t i = 0; i < sorted.size() && i < count; i++) {
			text += fmt::format(" {}x{}", sorted[i].first, sorted[i].second);
		}
		return text;
	}

	void MaybeSummarize() {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_last_summary < std::chrono::seconds(60)) return;
		m_last_summary = now;
		std::vector<std::pair<uint64_t, const ShaderStats*>> sorted;
		std::map<std::string, uint32_t>                       all_reasons;
		uint32_t                                              total = 0;
		for (const auto& [hash, stats]: m_shaders) {
			sorted.emplace_back(hash, &stats);
			total += stats.variants;
			for (const auto& [reason, count]: stats.reasons) {
				// Per-slot reasons are pooled by field: "image3.mip_count" -> "image.mip_count".
				std::string pooled = reason;
				const auto  dot    = pooled.find('.');
				if (dot != std::string::npos) {
					auto start = dot;
					while (start > 0 && std::isdigit(static_cast<unsigned char>(pooled[start - 1]))) start--;
					pooled.erase(start, dot - start);
				}
				all_reasons[pooled] += count;
			}
		}
		std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second->variants > b.second->variants; });
		std::string text = fmt::format("variant summary: {} shader compiles over {} shaders, reasons:{}\n", total,
		                               m_shaders.size(), TopReasons(all_reasons, 12));
		text += fmt::format("  {} pipelines, reasons:{}\n", m_pipelines, TopReasons(m_pipeline_reasons, 12));
		for (size_t i = 0; i < sorted.size() && i < 15; i++) {
			text += fmt::format("  {} 0x{:016x} variants={}{}\n", static_cast<int>(sorted[i].second->stage),
			                    sorted[i].first, sorted[i].second->variants,
			                    TopReasons(sorted[i].second->reasons, 4));
		}
		Log::WriteToConsoleAndLog(text);
	}

	bool                                       m_enabled = std::getenv("KYTY_NO_VARIANT_LOG") == nullptr;
	std::mutex                                 m_mutex;
	std::unordered_map<uint64_t, ShaderStats>  m_shaders;
	std::map<std::string, uint32_t>            m_pipeline_reasons;
	uint32_t                                   m_pipelines     = 0;
	uint32_t                                   m_shader_lines   = 0;
	uint32_t                                   m_pipeline_lines = 0;
	std::chrono::steady_clock::time_point      m_last_summary = std::chrono::steady_clock::now();
};

// The specialization fields that differ between an existing variant and the one being compiled.
void DiffSpecialization(const ShaderRecompiler::IR::ResourceSpecialization& from,
                        const ShaderRecompiler::IR::ResourceSpecialization& to,
                        std::vector<std::string>&                           reasons) {
	const auto field = [&reasons](const char* kind, size_t slot, const char* name, auto a, auto b) {
		if (a != b) {
			reasons.push_back(fmt::format("{}{}.{}({}->{})", kind, slot, name,
			                              static_cast<uint64_t>(a), static_cast<uint64_t>(b)));
		}
	};
	if (from.buffers.size() != to.buffers.size()) reasons.emplace_back("buffer-count");
	for (size_t i = 0; i < std::min(from.buffers.size(), to.buffers.size()); i++) {
		const auto& a = from.buffers[i];
		const auto& b = to.buffers[i];
		field("buffer", i, "stride", a.packed_stride, b.packed_stride);
		field("buffer", i, "format", a.descriptor_format, b.descriptor_format);
		field("buffer", i, "swizzle", a.descriptor_swizzle, b.descriptor_swizzle);
		field("buffer", i, "zero_stride_oob", a.zero_stride_oob, b.zero_stride_oob);
		field("buffer", i, "indirect_root", a.indirect_root, b.indirect_root);
		field("buffer", i, "indirect_offset", a.indirect_mapping_offset, b.indirect_mapping_offset);
		field("buffer", i, "indirect_search", a.indirect_search_iterations, b.indirect_search_iterations);
	}
	if (from.images.size() != to.images.size()) reasons.emplace_back("image-count");
	for (size_t i = 0; i < std::min(from.images.size(), to.images.size()); i++) {
		const auto& a = from.images[i];
		const auto& b = to.images[i];
		field("image", i, "numeric_class", a.numeric_class, b.numeric_class);
		field("image", i, "dimension", a.dimension, b.dimension);
		field("image", i, "mip_count", a.mip_count, b.mip_count);
		field("image", i, "conversion_format", a.conversion_format, b.conversion_format);
		field("image", i, "swizzle", a.shader_swizzle, b.shader_swizzle);
		field("image", i, "indirect_root", a.indirect_root, b.indirect_root);
		field("image", i, "indirect_offset", a.indirect_mapping_offset, b.indirect_mapping_offset);
		field("image", i, "indirect_search", a.indirect_search_iterations, b.indirect_search_iterations);
		field("image", i, "cube", a.cube, b.cube);
		field("image", i, "fmask", a.fmask, b.fmask);
	}
}

uint64_t SpecializationHash(const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	uint64_t   hash = 0;
	const auto mix  = [&hash](uint64_t value) { hash = XXH3_64bits_withSeed(&value, sizeof(value), hash); };
	for (const auto& b: specialization.buffers) {
		mix(b.packed_stride);
		mix(static_cast<uint64_t>(b.descriptor_format));
		mix(b.descriptor_swizzle);
		mix(b.zero_stride_oob ? 1u : 0u);
		mix(b.indirect_root);
		mix(b.indirect_mapping_offset);
		mix(b.indirect_search_iterations);
	}
	mix(0xffffffffu);
	for (const auto& i: specialization.images) {
		mix(static_cast<uint64_t>(i.numeric_class));
		mix(static_cast<uint64_t>(i.dimension));
		mix(i.mip_count);
		mix(static_cast<uint64_t>(i.conversion_format));
		mix(i.shader_swizzle);
		mix(i.indirect_root);
		mix(i.indirect_mapping_offset);
		mix(i.indirect_search_iterations);
		mix((i.cube ? 1u : 0u) | (i.fmask ? 2u : 0u));
	}
	return hash;
}

// One line per compiled program in _gt7_spirv_<time>.txt: the same shader, specialization and
// push-data layout should give the same SPIR-V hash in every run.
void WriteSpirvManifest(const ShaderRecompiler::CompileOptions&             options,
                        const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                        uint32_t push_data_start, const std::vector<uint32_t>& spirv) {
	static std::mutex mutex;
	static FILE*      file = [] {
		const auto now = std::time(nullptr);
		std::tm    local {};
		localtime_s(&local, &now);
		char name[64] = {};
		std::strftime(name, sizeof(name), "_gt7_spirv_%Y%m%d_%H%M%S.txt", &local);
		return std::fopen(name, "w");
	}();
	if (file == nullptr) return;
	const auto spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
	std::scoped_lock lock(mutex);
	std::fprintf(file, "%d %016llx %u %016llx %016llx %zu\n", static_cast<int>(options.stage),
	             static_cast<unsigned long long>(options.shader_hash), push_data_start,
	             static_cast<unsigned long long>(SpecializationHash(specialization)),
	             static_cast<unsigned long long>(spirv_hash), spirv.size());
	std::fflush(file);
	// KYTY_DUMP_SPIRV=hash,...: the SPIR-V of those shaders, one file per specialization and
	// run, to compare runs.
	static const std::vector<uint64_t> dump = [] {
		std::vector<uint64_t> list;
		if (const char* env = std::getenv("KYTY_DUMP_SPIRV"); env != nullptr) {
			const std::string text(env);
			for (size_t start = 0; start < text.size();) {
				const auto end = std::min(text.find(',', start), text.size());
				if (end > start) list.push_back(std::stoull(text.substr(start, end - start), nullptr, 16));
				start = end + 1;
			}
		}
		return list;
	}();
	if (std::ranges::find(dump, options.shader_hash) != dump.end()) {
		static const auto stamp = std::time(nullptr);
		const auto        name  = fmt::format("_spirv_dumps/{:016x}_{:016x}_{}_{}.spv", options.shader_hash,
		                                      SpecializationHash(specialization), push_data_start, stamp);
		std::filesystem::create_directories("_spirv_dumps");
		if (FILE* out = std::fopen(name.c_str(), "wb"); out != nullptr) {
			std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), out);
			std::fclose(out);
		}
	}
}

} // namespace

bool TakeSkippedDraw() {
	return std::exchange(g_skip_draw, false);
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		template <bool HasCalls>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor) {
			EXIT_IF(!MaterializeFresh(resource_plan, runtime, resources, specialization));
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque keeps permutations in place: background pipeline jobs hold their program info.
		std::deque<Permutation>                     permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	using ProgramMap = std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>;

	// The recorded permutations of one source key. Workers compile queued jobs; the GPU thread
	// takes a job when it first needs its key, waiting for it or compiling it itself.
	struct Job {
		enum class State { Dormant, Queued, Running, Done, Taken };

		State                      state = State::Dormant;
		ShaderCacheRecipes         recipes;
		std::optional<SourceEntry> result;
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;
	static constexpr auto        PeriodicSaveInterval = std::chrono::seconds(60);

	struct StageNames {
		const char* label = nullptr;
		const char* name  = nullptr;
	};

	static StageNames GetStageNames(ShaderType stage) {
		switch (stage) {
			case ShaderType::Vertex: return {"ShaderRecompiler VS", "vs"};
			case ShaderType::Mesh: return {"ShaderRecompiler MS", "ms"};
			case ShaderType::Local: return {"ShaderRecompiler LS", "ls"};
			case ShaderType::TessellationControl: return {"ShaderRecompiler HS", "hs"};
			case ShaderType::TessellationEvaluation: return {"ShaderRecompiler DS", "ds"};
			case ShaderType::Pixel: return {"ShaderRecompiler PS", "ps"};
			case ShaderType::Compute: return {"ShaderRecompiler CS", "cs"};
			default: EXIT("invalid pipeline shader stage\n");
		}
		return {};
	}

	template <typename InputInfo>
	static ShaderType GetStage(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			return ShaderType::Compute;
		}
	}

	// The same options for a draw and for a recorded recipe; `input_info` must outlive them.
	template <typename InputInfo>
	ShaderRecompiler::CompileOptions
	MakeOptions(uint64_t hash, std::span<const uint32_t> user_data,
	            std::span<const uint32_t> back_code, const InputInfo& input_info) const {
		const auto                       stage = GetStage(input_info);
		ShaderRecompiler::CompileOptions options;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.input_info.vertex = &input_info;
			options.user_data_base    = 8;
			options.wave_size         = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			options.input_info.pixel = &input_info;
			options.wave_size        = input_info.wave_size;
		} else {
			options.input_info.compute = &input_info;
			options.wave_size          = input_info.wave_size;
		}
		options.stage       = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash = hash;
		options.user_data   = user_data;
		options.back_code   = back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = GetStageNames(stage).label;
		return options;
	}

	// Safe on any thread: the GPU thread numbers the program when it adopts it.
	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t                  push_data_start_dword,
	                               std::span<const uint32_t> function_code) const {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		const auto loop_cap = ShaderRecompiler::LoopCapFor(options.shader_hash);
		ShaderRecompiler::SetLoopCapOverride(loop_cap);
		if (loop_cap != 0) {
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::scoped_lock                    lock(logged_mutex);
			if (logged.insert(options.shader_hash).second) {
				Log::WriteToConsoleAndLog(fmt::format("watchdog: {} 0x{:016x} loops capped at {}\n",
				                                      stage_name, options.shader_hash, loop_cap));
			}
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		WriteSpirvManifest(options, specialization, push_data_start_dword, result.spirv);
		if (const auto guards = ShaderRecompiler::LastWaterfallLoopGuards(); guards != 0) {
			Log::WriteToConsoleAndLog(fmt::format("watchdog: {} 0x{:016x} waterfall loop exits capped ({})\n",
			                                      stage_name, options.shader_hash, guards));
		}

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id         = 0,
		                       .module     = module,
		                       .spirv_hash = XXH3_64bits(result.spirv.data(),
		                                                 result.spirv.size() * sizeof(uint32_t))},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		CheckLookupThread();
		const auto stage           = GetStage(input_info);

		DebugNoteShader(params.hash, static_cast<uint32_t>(stage));
		if (stage == ShaderType::Pixel && ShaderRecompiler::SkipRuntimeCallShader(params.hash)) {
			g_skip_draw = true;
		}
		if (stage != ShaderType::Compute && DumpShaderInputs(params.hash)) {
			static std::atomic_int dumps = 0;
			if (dumps++ < 20) {
				auto text = fmt::format("lookup {} 0x{:016x} user_data:", static_cast<int>(stage),
				                        params.hash);
				for (uint32_t i = 0; i < params.user_data_count; i++) {
					text += fmt::format(" s{}={:08x}", i, params.user_data[i]);
				}
				const uint64_t table = params.user_data_count >= 2
				                           ? ((uint64_t(params.user_data[1] & 0xffffu) << 32u) |
				                              params.user_data[0]) &
				                                 ~uint64_t {3}
				                           : 0;
				std::array<uint32_t, 320> words {};
				if (table != 0 && Libs::LibKernel::Memory::TryReadBacking(table, words.data(),
				                                                          sizeof(words))) {
					text += fmt::format("\n  table 0x{:x}:", table);
					for (uint32_t i = 0; i < words.size(); i++) {
						if (i % 8 == 0) text += fmt::format("\n   +{:4}:", i * 4);
						text += fmt::format(" {:08x}", words[i]);
					}
				}
				Log::WriteToConsoleAndLog(text + "\n");
			}
		}
		g_sync_before_shader_read = SpecializationSyncEnabled() || SerializeShader(params.hash);
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto entry = programs.find(lookup_key);
		if (entry == programs.end()) {
			entry = Adopt(lookup_key);
		}
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    // Experimental: ordinary scalar reads also wait for GPU-written bytes.
		    .read_memory                = ReadShaderGuestMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			Permutation* permutation = nullptr;
			if (cached.call_source) {
				source_changed = cached.RefreshCallSource(runtime);
				if (!source_changed) permutation = cached.FindPermutation<true>(runtime, push_data_cursor);
			} else {
				permutation = cached.FindPermutation<false>(runtime, push_data_cursor);
			}
			if (permutation != nullptr) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		if (VariantLog::Get().Enabled()) {
			NoteVariantReasons(entry, source_changed, push_data_cursor);
		}
		PerfScope  compile_time(PerfCounter::CommandThreadCompile);
		const auto options    = MakeOptions(params.hash, user_data, params.back_code, input_info);
		const auto stage_name = GetStageNames(stage).name;
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (!current->calls.empty()) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			auto retained = !source.calls.empty() ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			EXIT_IF(!MaterializeFresh(entry->second.resource_plan, runtime,
			                          entry->second.resources, entry->second.specialization));
		} else if (source_changed) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (const auto* permutation = entry->second.FindPermutation<true>(runtime, push_data_cursor)) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor,
		    entry->second.FunctionCode()));
		auto& permutation     = entry->second.permutations.back();
		permutation.handle.id = ++next_shader_id;
		// Only programs that compiled are recorded, so a cache never fails earlier than a draw.
		// A SWAPPC program also depends on the guest functions it links, which are not recorded.
		if (!entry->second.call_source) {
			Record(params, input_info, permutation.specialization, push_data_cursor);
		}
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		{
			std::lock_guard lock(job_mutex);
			stats.compiled++;
		}

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// Records why the shader in `lookup_key` needs a new variant (see VariantLog). For a new key
	// the existing key of the same shader shows which static state changed; for a known key, the
	// closest recorded variant shows which specialization fields changed.
	void NoteVariantReasons(ProgramMap::iterator entry, bool source_changed,
	                        uint32_t push_data_cursor) {
		std::vector<std::string> reasons;
		if (entry == programs.end()) {
			const ProgramKey* sibling = nullptr;
			for (const auto& [key, source]: programs) {
				if (key.stage == lookup_key.stage && key.hash == lookup_key.hash) {
					sibling = &key;
					break;
				}
			}
			if (sibling == nullptr) {
				reasons.emplace_back("first-use");
			} else if (sibling->user_data_count != lookup_key.user_data_count ||
			           sibling->code_size != lookup_key.code_size) {
				reasons.emplace_back("code-or-user-data-size");
			} else if (sibling->static_state.size() != lookup_key.static_state.size()) {
				reasons.emplace_back("static-state-size");
			} else {
				for (size_t i = 0; i < sibling->static_state.size() && reasons.size() < 4; i++) {
					if (sibling->static_state[i] != lookup_key.static_state[i]) {
						reasons.push_back(fmt::format("static-word{}({:x}->{:x})", i,
						                              sibling->static_state[i],
						                              lookup_key.static_state[i]));
					}
				}
			}
		} else if (source_changed) {
			reasons.emplace_back("call-target-code-changed");
		} else if (entry->second.permutations.empty()) {
			reasons.emplace_back("first-use");
		} else {
			const auto&              current = entry->second;
			std::vector<std::string> best;
			bool                     have_best = false;
			for (const auto& candidate: current.permutations) {
				std::vector<std::string> diff;
				const auto& layout = candidate.program.bindings;
				const auto  start  = ShaderRecompiler::IR::PushData::StartFor(
                    push_data_cursor, layout.ShaderDataDwords());
				if (layout.push_data_start_dword != start) {
					diff.push_back(fmt::format("push-data-start({}->{})", layout.push_data_start_dword, start));
				}
				DiffSpecialization(candidate.specialization, current.specialization, diff);
				if (current.call_source && !std::ranges::equal(candidate.function_code, current.FunctionCode())) {
					diff.emplace_back("call-target-code");
				}
				if (!have_best || diff.size() < best.size()) {
					best      = std::move(diff);
					have_best = true;
				}
			}
			reasons = best.empty() ? std::vector<std::string> {"unknown"} : std::move(best);
			if (reasons.size() > 6) {
				reasons.resize(6);
				reasons.emplace_back("...");
			}
		}
		VariantLog::Get().NoteShader(lookup_key.hash, lookup_key.stage, reasons);
	}

	// `programs` and `lookup_key` belong to the one thread that looks programs up (the GPU thread);
	// workers only touch jobs, under job_mutex.
	void CheckLookupThread() {
		const auto      current = std::this_thread::get_id();
		std::thread::id owner;
		EXIT_IF(!lookup_thread.compare_exchange_strong(owner, current) && owner != current);
	}

	// Moves the background result for `key` into the cache. Without one, marks the key so that a
	// later prefetch does not compile it a second time.
	ProgramMap::iterator Adopt(const ProgramKey& key) {
		std::optional<SourceEntry> result;
		{
			std::unique_lock lock(job_mutex);
			auto [iter, inserted] = jobs.try_emplace(key);
			auto& job             = iter->second;
			if (inserted) {
				job.state = Job::State::Taken;
				NoteRecipeMiss(key);
				return programs.end();
			}
			switch (job.state) {
				case Job::State::Taken: return programs.end();
				case Job::State::Running: {
					const auto begin = std::chrono::steady_clock::now();
					WaitForJob(lock, job);
					stats.waited++;
					stats.waited_us +=
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
					                              std::chrono::steady_clock::now() - begin)
					                              .count());
					result = std::move(job.result);
					break;
				}
				case Job::State::Done: result = std::move(job.result); break;
				case Job::State::Dormant:
				case Job::State::Queued: {
					// Not started yet: compiling here is faster than waiting behind other jobs.
					std::erase(queue, &job);
					job.state = Job::State::Running;
					running++;
					const auto hook = compile_hook;
					lock.unlock();
					result = CompileJob(job.recipes, hook);
					lock.lock();
					stats.compiled += result ? result->permutations.size() : 0;
					// Ends like a worker job, so that every waiter sees it finish.
					job.state = Job::State::Done;
					running--;
					job_done.notify_all();
					break;
				}
			}
			job.state = Job::State::Taken;
			job.result.reset();
			job.recipes = {};
		}
		if (!result) {
			return programs.end();
		}
		for (auto& permutation: result->permutations) {
			permutation.handle.id = ++next_shader_id;
		}
		return programs.emplace(key, std::move(*result)).first;
	}

	// A key with no recorded recipe, while recipes of the same shader exist: logs how the key
	// differs from the closest one, to find state that changes between runs. Under job_mutex.
	void NoteRecipeMiss(const ProgramKey& key) {
		if (!VariantLog::Get().Enabled()) return;
		static uint32_t logged = 0;
		const ProgramKey* closest = nullptr;
		size_t            closest_diffs = SIZE_MAX;
		for (const auto& [other, job]: jobs) {
			if (other.stage != key.stage || other.hash != key.hash || &other == &key ||
			    other == key || job.state == Job::State::Taken) {
				continue;
			}
			size_t diffs = (other.user_data_count != key.user_data_count) +
			               (other.code_size != key.code_size) +
			               (other.static_state.size() != key.static_state.size()) * 100u;
			for (size_t i = 0; i < std::min(other.static_state.size(), key.static_state.size()); i++) {
				diffs += other.static_state[i] != key.static_state[i];
			}
			if (diffs < closest_diffs) {
				closest       = &other;
				closest_diffs = diffs;
			}
		}
		if (closest == nullptr || logged >= 300) return;
		logged++;
		std::string text = fmt::format("recipe miss: {} 0x{:016x}:", static_cast<int>(key.stage), key.hash);
		if (closest->user_data_count != key.user_data_count) {
			text += fmt::format(" user_data_count({}->{})", closest->user_data_count, key.user_data_count);
		}
		if (closest->code_size != key.code_size) {
			text += fmt::format(" code_size({}->{})", closest->code_size, key.code_size);
		}
		if (closest->static_state.size() != key.static_state.size()) {
			text += fmt::format(" static_size({}->{})", closest->static_state.size(), key.static_state.size());
		}
		uint32_t shown = 0;
		for (size_t i = 0; i < std::min(closest->static_state.size(), key.static_state.size()); i++) {
			if (closest->static_state[i] != key.static_state[i] && shown++ < 8) {
				text += fmt::format(" word{}({:x}->{:x})", i, closest->static_state[i], key.static_state[i]);
			}
		}
		Log::WriteToConsoleAndLog(text + "\n");
	}

	// Wakes once the job has ended, whether its result is still there (Done) or adopted (Taken).
	void WaitForJob(std::unique_lock<std::mutex>& lock, const Job& job) {
		job_done.wait(lock, [&job] { return job.state != Job::State::Running; });
	}

	template <typename InputInfo>
	std::optional<Permutation>
	CompileRecipe(const ShaderCacheRecipe&                           recipe,
	              std::optional<ShaderRecompiler::IR::ResourcePlan>& plan) const {
		InputInfo input_info;
		EXIT_IF(!DecodeShaderCacheInput(recipe.input, input_info)); // Checked by RecipeKey.
		// Translation reads the user data count, not the values.
		const std::vector<uint32_t> user_data(recipe.user_data_count, 0u);
		const auto options    = MakeOptions(recipe.hash, user_data, recipe.back_code, input_info);
		const auto stage_name = GetStageNames(recipe.stage).name;
		DumpShaderOriginal(stage_name, recipe.hash, recipe.code);
		const auto source = ShaderRecompiler::PrepareShaderSource(recipe.code, options);
		// A SWAPPC program links guest functions that the record does not hold.
		if (!source.calls.empty()) {
			return std::nullopt;
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		if (!plan) {
			plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
		}
		return CompilePermutation(stage_name, options, std::move(translated), recipe.specialization,
		                          recipe.push_data_start, {});
	}

	// Compiles every recipe of one source key; safe on any thread.
	std::optional<SourceEntry> CompileJob(const ShaderCacheRecipes&    recipes,
	                                      const std::function<void()>& hook) const {
		if (hook) {
			hook();
		}
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		std::vector<Permutation>                          permutations;
		for (const auto& recipe: recipes) {
			std::optional<Permutation> permutation;
			switch (recipe->stage) {
				case ShaderType::Pixel:
					permutation = CompileRecipe<ShaderPixelInputInfo>(*recipe, plan);
					break;
				case ShaderType::Compute:
					permutation = CompileRecipe<ShaderComputeInputInfo>(*recipe, plan);
					break;
				default: permutation = CompileRecipe<ShaderVertexInputInfo>(*recipe, plan); break;
			}
			// Determinism check: the same recipe must give the same SPIR-V, or the driver's
			// pipeline cache can never match a pipeline from an earlier run.
			static std::atomic<int> checks_left {200};
			if (permutation && checks_left.fetch_sub(1, std::memory_order_relaxed) > 0) {
				std::optional<ShaderRecompiler::IR::ResourcePlan> scratch_plan;
				std::optional<Permutation>                        again;
				switch (recipe->stage) {
					case ShaderType::Pixel:
						again = CompileRecipe<ShaderPixelInputInfo>(*recipe, scratch_plan);
						break;
					case ShaderType::Compute:
						again = CompileRecipe<ShaderComputeInputInfo>(*recipe, scratch_plan);
						break;
					default: again = CompileRecipe<ShaderVertexInputInfo>(*recipe, scratch_plan); break;
				}
				if (again) {
					if (again->handle.spirv_hash != permutation->handle.spirv_hash) {
						Log::WriteToConsoleAndLog(fmt::format(
						    "determinism: {} 0x{:016x} compiled twice gives different SPIR-V\n",
						    static_cast<int>(recipe->stage), recipe->hash));
					}
					device.destroyShaderModule(again->handle.module, nullptr);
				}
			}
			if (!permutation) {
				// The draw compiles the key itself.
				for (const auto& compiled: permutations) {
					device.destroyShaderModule(compiled.handle.module, nullptr);
				}
				return std::nullopt;
			}
			permutations.push_back(std::move(*permutation));
		}
		if (!plan) {
			return std::nullopt;
		}
		std::optional<SourceEntry> entry;
		entry.emplace(std::move(*plan), nullptr);
		entry->permutations.assign(std::make_move_iterator(permutations.begin()),
		                          std::make_move_iterator(permutations.end()));
		return entry;
	}

	// The key a recipe compiles to, or nothing for input that this build cannot use.
	static std::optional<ProgramKey> RecipeKey(const ShaderCacheRecipe& recipe) {
		ProgramKey key {
		    .stage           = recipe.stage,
		    .hash            = recipe.hash,
		    .user_data_count = recipe.user_data_count,
		    .code_size       = static_cast<uint32_t>(recipe.code.size()),
		};
		switch (recipe.stage) {
			case ShaderType::Pixel: {
				ShaderPixelInputInfo info;
				if (!DecodeShaderCacheInput(recipe.input, info) ||
				    info.input_num > std::size(info.interpolator_settings)) {
					return std::nullopt;
				}
				BuildStageStaticKey(info, key.static_state);
				break;
			}
			case ShaderType::Compute: {
				ShaderComputeInputInfo info;
				if (!DecodeShaderCacheInput(recipe.input, info)) {
					return std::nullopt;
				}
				BuildStageStaticKey(info, key.static_state);
				break;
			}
			default: {
				ShaderVertexInputInfo info;
				if (!DecodeShaderCacheInput(recipe.input, info) ||
				    info.logical_stage != recipe.stage || info.resources_num < 0 ||
				    info.resources_num > ShaderVertexInputInfo::RES_MAX) {
					return std::nullopt;
				}
				BuildStageStaticKey(info, key.static_state);
				break;
			}
		}
		return key;
	}

	template <typename InputInfo>
	void Record(const ShaderParams& params, const InputInfo& input_info,
	            const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	            uint32_t                                            push_data_start) {
		if (cache_path.empty()) {
			return;
		}
		auto recipe             = std::make_shared<ShaderCacheRecipe>();
		recipe->stage           = GetStage(input_info);
		recipe->user_data_count = params.user_data_count;
		recipe->push_data_start = push_data_start;
		recipe->hash            = params.hash;
		recipe->code.assign(params.code.begin(), params.code.end());
		recipe->back_code.assign(params.back_code.begin(), params.back_code.end());
		recipe->input          = EncodeShaderCacheInput(input_info);
		recipe->specialization = specialization;
		std::lock_guard lock(job_mutex);
		recipes.push_back(std::move(recipe));
		dirty = true;
		if (std::chrono::steady_clock::now() - last_save > PeriodicSaveInterval) {
			last_save      = std::chrono::steady_clock::now();
			save_requested = true;
			StartWorkers();
			job_ready.notify_one();
		}
	}

	void Load(const std::filesystem::path& path, std::string signature) {
		cache_path      = path;
		cache_signature = std::move(signature);
		std::vector<uint8_t> data;
		ShaderCacheRecipes   loaded;
		size_t               parsed    = 0;
		const auto           path_text = Common::PathToString(path);
		if (ReadShaderCacheFile(path, data) &&
		    !ParseShaderCache(data, cache_signature, loaded, &parsed)) {
			PipelineCacheLog(
			    "Shader cache: invalidating {} (graphics code, device or driver changed)",
			    path_text);
		}
		{
			std::lock_guard lock(job_mutex);
			for (auto& recipe: loaded) {
				auto key = RecipeKey(*recipe);
				if (!key) {
					continue;
				}
				auto& job     = jobs[std::move(*key)];
				auto& trigger = triggers[XXH3_64bits(recipe->code.data(),
				                                     recipe->code.size() * sizeof(uint32_t))];
				if (std::ranges::find(trigger, &job) == trigger.end()) {
					trigger.push_back(&job);
				}
				job.recipes.push_back(recipe);
				recipes.push_back(std::move(recipe));
			}
			// The next save drops a damaged tail, unusable records or a stale file.
			dirty     = parsed != data.size() || recipes.size() != loaded.size();
			last_save = std::chrono::steady_clock::now();
			PipelineCacheLog("Shader cache: {} recorded programs in {} sources from {}",
			                 recipes.size(), jobs.size(), path_text);
		}
		ShaderAddMapObserver(MapObserver, this);
	}

	static void MapObserver(void* user, uint64_t hash) {
		static_cast<ProgramCache*>(user)->OnShaderMapped(hash);
	}

	// The game loaded a shader: compile its recorded programs before the first draw needs them.
	void OnShaderMapped(uint64_t hash) {
		std::lock_guard lock(job_mutex);
		const auto      trigger = triggers.find(hash);
		if (trigger == triggers.end()) {
			return;
		}
		for (auto* job: trigger->second) {
			if (job->state == Job::State::Dormant) {
				job->state = Job::State::Queued;
				queue.push_back(job);
			}
		}
		triggers.erase(trigger);
		StartWorkers();
		job_ready.notify_all();
	}

	// Called with job_mutex held.
	void StartWorkers() {
		if (!workers.empty()) {
			return;
		}
		const auto count = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 8u);
		for (uint32_t i = 0; i < count; i++) {
			workers.emplace_back([this](std::stop_token stop) { WorkerLoop(stop); });
		}
	}

	void WorkerLoop(const std::stop_token& stop) {
		std::unique_lock lock(job_mutex);
		for (;;) {
			job_ready.wait(lock, [this, &stop] {
				return stop.stop_requested() || !queue.empty() || save_requested;
			});
			if (stop.stop_requested()) {
				return;
			}
			if (save_requested) {
				save_requested = false;
				lock.unlock();
				SaveRecipes();
				lock.lock();
				continue;
			}
			auto* job = queue.front();
			queue.pop_front();
			job->state = Job::State::Running;
			running++;
			const auto hook = compile_hook;
			lock.unlock();
			auto result = CompileJob(job->recipes, hook);
			lock.lock();
			stats.compiled_ahead += result ? result->permutations.size() : 0;
			job->result = std::move(result);
			job->state  = Job::State::Done;
			running--;
			job_done.notify_all();
		}
	}

	void SaveRecipes() {
		std::lock_guard    save_lock(save_mutex);
		ShaderCacheRecipes snapshot;
		{
			std::lock_guard lock(job_mutex);
			if (!dirty || cache_path.empty()) {
				return;
			}
			snapshot = recipes;
			dirty    = false;
		}
		const auto data = SerializeShaderCache(cache_signature, snapshot);
		if (!WriteShaderCacheFile(cache_path, data)) {
			std::lock_guard lock(job_mutex);
			dirty = true;
			PipelineCacheLog("Shader cache: failed to write {}", Common::PathToString(cache_path));
			return;
		}
		PipelineCacheLog("Shader cache: saved {} recorded programs ({} bytes) to {}",
		                 snapshot.size(), data.size(), Common::PathToString(cache_path));
	}

	void WaitForBackgroundCompilation() {
		std::unique_lock lock(job_mutex);
		const auto       finished = [this] { return queue.empty() && running == 0; };
		// Under job_mutex: no job can end between the hook and the wait.
		if (wait_hook) {
			wait_hook(!finished());
		}
		job_done.wait(lock, finished);
	}

	explicit ProgramCache(GraphicContext& graphics): device(graphics.device), graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		ShaderRemoveMapObserver(MapObserver, this);
		{
			std::lock_guard lock(job_mutex);
			// Jobs that never started stay dormant; running jobs end as Done and wake waiters.
			for (auto* job: queue) {
				job->state = Job::State::Dormant;
			}
			queue.clear();
			for (auto& worker: workers) {
				worker.request_stop();
			}
		}
		job_ready.notify_all();
		job_done.notify_all();
		workers.clear();
		if (!cache_path.empty()) {
			PipelineCacheLog("Shader cache: {} programs compiled when needed, {} compiled ahead, "
			                 "{} waits ({} ms)",
			                 stats.compiled, stats.compiled_ahead, stats.waited,
			                 stats.waited_us / 1000u);
		}
		auto destroy = [this](const SourceEntry& entry) {
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		};
		for (const auto& [key, entry]: programs) {
			destroy(entry);
		}
		for (const auto& [key, job]: jobs) {
			if (job.result) {
				destroy(*job.result);
			}
		}
	}

	ProgramMap                                                  programs;
	ProgramKey                                                  lookup_key;
	std::atomic<std::thread::id>                                lookup_thread;
	vk::Device                                                  device;
	GraphicContext&                                             graphics;
	uint64_t                                                    next_shader_id = 0;

	// Shared with the workers and the shader map observer, under job_mutex.
	std::mutex                                          job_mutex;
	std::condition_variable                             job_done;
	std::condition_variable                             job_ready;
	std::unordered_map<ProgramKey, Job, ProgramKeyHash> jobs;
	std::unordered_map<uint64_t, std::vector<Job*>>     triggers;
	std::deque<Job*>                                    queue;
	uint32_t                                            running = 0;
	std::function<void()>                               compile_hook;
	std::function<void(bool)>                           wait_hook;
	ShaderCacheRecipes                                  recipes;
	bool                                                dirty          = false;
	bool                                                save_requested = false;
	std::chrono::steady_clock::time_point               last_save;
	ProgramStats                                        stats;
	std::mutex                                          save_mutex;
	std::filesystem::path                               cache_path;
	std::string                                         cache_signature;
	// Declared last: destroyed first, before the state the workers use.
	std::vector<std::jthread> workers;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeCaches("_PipelineCache", PipelineCacheTitleId());
}

PipelineCache::PipelineCache(GraphicContext& graphics, const std::filesystem::path& directory,
                             const std::string& title_id)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)) {
	InitializeCaches(directory, title_id);
}

PipelineCache::~PipelineCache() {
	StopPipelineWorkers();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeCaches(const std::filesystem::path& directory,
                                     const std::string&           title_id) {
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_program_cache->Load(directory / (title_id + ".shaders.bin"),
	                      // Development builds keep their recipes: each one is only compile
	                      // input, and lookups still check the key and specialization. Bump the
	                      // tag when the recipe format or specialization changes (upstream 5dec523
	                      // added samplers and scalar_empty: older recipes failed the check).
	                      CacheSignature("KytySC1", "gt7-dev-4",
	                                     m_graphics.GetPhysicalDeviceProperties()));
	InitializeDriverCache(directory / (title_id + ".bin"));
}

void PipelineCache::InitializeDriverCache(const std::filesystem::path& cache_path) {
	m_driver_cache_path     = cache_path;
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		// The driver cache only grows: past KYTY_PIPELINE_CACHE_MAX_MB (default 1536) it starts over,
		// as GT7's reached the 4 GiB save limit and every later save failed.
		static const uint64_t max_bytes = [] {
			const char* env = std::getenv("KYTY_PIPELINE_CACHE_MAX_MB");
			return (env != nullptr ? std::strtoull(env, nullptr, 10) : 1536ull) * 1024ull * 1024ull;
		}();
		if (file_size > max_bytes) {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: {} is {} MB, over the limit; starting empty", path,
			                 file_size >> 20u);
		} else if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	// Workers create pipelines through the driver cache that is released below.
	DrainPipelineJobs();
	m_program_cache->SaveRecipes();
	if (m_driver_cache == nullptr) {
		return;
	}
	WriteDriverCache();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::NotePipelineCreated() {
	// Save periodically: the final save only runs when the window closes normally.
	m_pipelines_since_save++;
	const auto now = std::chrono::steady_clock::now();
	if (m_pipelines_since_save >= 50 && now - m_last_save >= std::chrono::seconds(30)) {
		WriteDriverCache();
		m_pipelines_since_save = 0;
		m_last_save            = now;
	}
}

bool PipelineCache::WriteDriverCache() {
	if (m_driver_cache == nullptr) {
		return false;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	return true;
}

PipelineCache::ProgramStats PipelineCache::GetProgramStats() const {
	std::lock_guard lock(m_program_cache->job_mutex);
	auto            stats = m_program_cache->stats;
	stats.recorded        = m_program_cache->recipes.size();
	stats.in_flight       = m_program_cache->running;
	return stats;
}

void PipelineCache::WaitForBackgroundCompilation() {
	m_program_cache->WaitForBackgroundCompilation();
}

void PipelineCache::SetRecordedCompileHook(std::function<void()> hook) {
	std::lock_guard lock(m_program_cache->job_mutex);
	m_program_cache->compile_hook = std::move(hook);
}

void PipelineCache::SetBackgroundWaitHook(std::function<void(bool blocking)> hook) {
	std::lock_guard lock(m_program_cache->job_mutex);
	m_program_cache->wait_hook = std::move(hook);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (!Prospero::IsPackedUnormTargetFormat(format.buffer_format)) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			const uint32_t all = (1u << format.components) - 1u;
			const auto mode = pixel_info.target_output_mode[export_index];
			// Float exports: 32-bit (9), or FP16/UNorm16/SNorm16 (4/5/6), unpacked to floats first.
			if (blend || (mapping.ApplyMask(target_mask) & all) != all ||
			    (mapping.ApplyMask(shader_mask) & all) != all ||
			    (mode != 9u && mode != 4u && mode != 5u && mode != 6u)) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			if (format.buffer_format == Prospero::BufferFormat::k2_10_10_10UNorm) {
				pixel_info.target_packed_unorm_10_mask |= 1u << export_index;
			}
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id);
	GraphicsPrograms  result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (Prospero::IsPackedUnormTargetFormat(colors[i].desc.info.guest_format)) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples < depth.desc.info.samples &&
		           m_graphics.mixed_attachment_samples_enabled) {
			// EQAA passes (GT7 since upstream 5dec523): fewer colour fragments than depth samples.
			// Rasterize at the depth sample count; colour attachments keep theirs (mixed samples).
			static_params.color_samples = static_cast<uint8_t>(attachment_samples);
			attachment_samples          = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			// Without mixed-sample support, skip such draws (KYTY_SKIP_MIXED_SAMPLE_DRAWS=0 restores
			// the fatal error).
			static LiveSetting skip_mixed("KYTY_SKIP_MIXED_SAMPLE_DRAWS", 1);
			if (skip_mixed.Get() == 0) {
				EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
				     depth.desc.info.samples);
			}
			static std::atomic<uint32_t> skipped {0};
			if (skipped.fetch_add(1, std::memory_order_relaxed) < 8) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: skipping a draw with mixed color/depth sample counts ({} and {})\n",
				    attachment_samples, depth.desc.info.samples));
			}
			return nullptr;
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return iter->second->ready.load(std::memory_order_acquire) ? iter->second.get() : nullptr;
	}

	if (VariantLog::Get().Enabled()) {
		std::vector<std::string>   reasons;
		const GraphicsPipelineKey* same_programs = nullptr;
		bool                       same_vs = false, same_ps = false;
		for (const auto& [other, pipeline]: m_graphics_pipelines) {
			const bool vs = other.vertex_shader_ids == key.vertex_shader_ids;
			const bool ps = other.ps_shader_id == key.ps_shader_id;
			same_vs |= vs;
			same_ps |= ps;
			if (vs && ps) {
				same_programs = &other;
				break;
			}
		}
		if (same_programs == nullptr) {
			reasons.emplace_back(!same_vs && !same_ps ? "new-vs+ps" : !same_vs ? "new-vs" : "new-ps");
		} else {
			const auto& other = *same_programs;
			if (!(other.rendering == key.rendering)) {
				if (other.rendering.color_count != key.rendering.color_count) reasons.emplace_back("color-count");
				else if (other.rendering.color_formats != key.rendering.color_formats) reasons.emplace_back("color-format");
				if (other.rendering.depth_format != key.rendering.depth_format ||
				    other.rendering.stencil_format != key.rendering.stencil_format) {
					reasons.emplace_back("depth-format");
				}
			}
			if (!(other.vertex_input == key.vertex_input)) {
				bool stride = other.vertex_input.binding_count != key.vertex_input.binding_count;
				bool offset = other.vertex_input.attribute_count != key.vertex_input.attribute_count;
				for (size_t i = 0; i < key.vertex_input.bindings.size(); i++) {
					stride |= !(other.vertex_input.bindings[i] == key.vertex_input.bindings[i]);
					offset |= !(other.vertex_input.attributes[i] == key.vertex_input.attributes[i]);
				}
				if (stride) reasons.emplace_back("vertex-stride");
				if (offset) reasons.emplace_back("vertex-attribute-offset");
			}
			const auto& a = other.static_params;
			const auto& b = key.static_params;
			const auto  differs = [](const auto& x, const auto& y) { return std::memcmp(&x, &y, sizeof(x)) != 0; };
			if (a.topology != b.topology || a.primitive_restart_enable != b.primitive_restart_enable) reasons.emplace_back("topology");
			if (a.samples != b.samples || a.sample_shading_enable != b.sample_shading_enable) reasons.emplace_back("samples");
			if (differs(a.color_mask, b.color_mask)) reasons.emplace_back("color-mask");
			if (a.cull_front != b.cull_front || a.cull_back != b.cull_back || a.face != b.face ||
			    a.polygon_mode != b.polygon_mode || a.provoking_vtx_last != b.provoking_vtx_last) {
				reasons.emplace_back("raster");
			}
			if (a.negative_one_to_one != b.negative_one_to_one || a.depth_clip_enable != b.depth_clip_enable) reasons.emplace_back("clip");
			if (differs(a.blend_enable, b.blend_enable) || differs(a.color_srcblend, b.color_srcblend) ||
			    differs(a.color_destblend, b.color_destblend) || differs(a.color_comb_fcn, b.color_comb_fcn) ||
			    differs(a.alpha_srcblend, b.alpha_srcblend) || differs(a.alpha_destblend, b.alpha_destblend) ||
			    differs(a.alpha_comb_fcn, b.alpha_comb_fcn) || differs(a.separate_alpha_blend, b.separate_alpha_blend)) {
				reasons.emplace_back("blend");
			}
			if (reasons.empty()) reasons.emplace_back("unknown");
		}
		VariantLog::Get().NotePipeline(reasons);
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	// Background creation: the draw is skipped until a worker finishes the pipeline, instead of
	// stalling the GPU thread for the driver's compile. Small targets (lookup tables, data
	// passes) and depth-less 1-pixel work are created at once, since later passes read them.
	static const bool sync_pipelines = std::getenv("KYTY_SYNC_PIPELINES") != nullptr;
	bool small_target = colors.empty() && !depth.image_id;
	for (const auto& color: colors) {
		const auto& extent = color.desc.info.extent;
		small_target |= uint64_t {extent.width} * extent.height < 256u * 256u;
	}
	if (!sync_pipelines && !small_target) {
		struct Job {
			PipelineRenderingState             rendering;
			PipelineVertexInputState           vertex_input;
			std::vector<ShaderVertexInputInfo> vertex_info;
			std::optional<ShaderPixelInputInfo> ps_input_info;
			GraphicsPrograms                   programs;
			PipelineStaticParameters           static_params;
		};
		auto job = std::make_shared<Job>(Job {
		    .rendering     = rendering,
		    .vertex_input  = key.vertex_input,
		    .vertex_info   = {vertex_info.begin(), vertex_info.end()},
		    .ps_input_info = ps_input_info != nullptr ? std::optional(*ps_input_info) : std::nullopt,
		    .programs      = programs,
		    .static_params = static_params,
		});
		cached->ready.store(false, std::memory_order_relaxed);
		auto* target = cached.get();
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		QueuePipelineJob([this, job, target] {
			PerfScope pipeline_time(PerfCounter::PipelineCreate);
			CreatePipelineInternal(m_graphics, *target, job->rendering, job->vertex_input,
			                       job->vertex_info,
			                       job->ps_input_info ? &*job->ps_input_info : nullptr,
			                       job->programs, job->static_params, m_driver_cache);
			EXIT_NOT_IMPLEMENTED(target->pipeline == nullptr);
			EXIT_NOT_IMPLEMENTED(target->pipeline_layout == nullptr);
			target->ready.store(true, std::memory_order_release);
		});
		NotePipelineCreated();
		return nullptr;
	}
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	PerfScope pipeline_time(PerfCounter::PipelineCreate);
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
	                                         Common::DebugCounters::Counter::PipelineCreateNs,
	                                         Common::DebugCounters::Peak::PipelineCreateNs);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	NotePipelineCreated();

	return iter->second.get();
}

void PipelineCache::QueuePipelineJob(std::function<void()> job) {
	std::unique_lock lock(m_pipeline_jobs_mutex);
	if (m_pipeline_workers.empty()) {
		// The driver compiles each new pipeline on one thread (about a second for GT7's large
		// pixel shaders), so throughput scales with workers; they run below normal priority so
		// the game and GPU threads keep their time.
		const auto hardware = std::max(std::thread::hardware_concurrency(), 4u);
		const auto count    = std::clamp(hardware / 2u + 2u, 2u, 12u);
		for (uint32_t i = 0; i < count; i++) {
			m_pipeline_workers.emplace_back([this] {
				LowerCurrentThreadPriority();
				for (;;) {
					std::function<void()> next;
					{
						std::unique_lock worker_lock(m_pipeline_jobs_mutex);
						m_pipeline_jobs_cv.wait(worker_lock, [this] {
							return m_pipeline_workers_stop || !m_pipeline_jobs.empty();
						});
						if (m_pipeline_jobs.empty()) return;
						next = std::move(m_pipeline_jobs.front());
						m_pipeline_jobs.pop_front();
						m_pipeline_jobs_running++;
					}
					next();
					{
						std::scoped_lock worker_lock(m_pipeline_jobs_mutex);
						m_pipeline_jobs_running--;
					}
					m_pipeline_idle_cv.notify_all();
				}
			});
		}
	}
	m_pipeline_jobs.push_back(std::move(job));
	lock.unlock();
	m_pipeline_jobs_cv.notify_one();
}

void PipelineCache::DrainPipelineJobs() {
	std::unique_lock lock(m_pipeline_jobs_mutex);
	m_pipeline_idle_cv.wait(lock, [this] {
		return m_pipeline_jobs.empty() && m_pipeline_jobs_running == 0;
	});
}

void PipelineCache::StopPipelineWorkers() {
	{
		std::scoped_lock lock(m_pipeline_jobs_mutex);
		m_pipeline_workers_stop = true;
	}
	m_pipeline_jobs_cv.notify_all();
	for (auto& worker: m_pipeline_workers) {
		worker.join();
	}
	m_pipeline_workers.clear();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	PerfScope pipeline_time(PerfCounter::PipelineCreate);
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
	                                         Common::DebugCounters::Counter::PipelineCreateNs,
	                                         Common::DebugCounters::Peak::PipelineCreateNs);
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache,
	                       compute_program.spirv_hash);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	NotePipelineCreated();

	return *iter->second;
}
} // namespace Libs::Graphics
