#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rendererLock.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// DLSS reads and writes with stages and accesses of its own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

// Round 21: one line whenever what the game asks of its upscaler changes (the render size, the
// output size, its flags, or whether DLSS took it), at most 64 lines a run.
uint64_t g_pssr_ns[3] {};

uint64_t PssrNow() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void NotePssr(const Pssr::Request& request, uint32_t output_width, uint32_t output_height,
              bool dlss) {
	// Round 21b: and every 1024 requests one line with the render sizes of those requests, since
	// the game changes its render size all the time and the lines above run out.
	static uint32_t count      = 0;
	static uint32_t dlss_count = 0;
	static uint32_t min_height = UINT32_MAX;
	static uint32_t max_height = 0;
	static uint64_t pixels     = 0;
	count++;
	dlss_count += dlss ? 1u : 0u;
	min_height = (std::min)(min_height, request.height);
	max_height = (std::max)(max_height, request.height);
	pixels += static_cast<uint64_t>(request.width) * request.height;
	if (count == 1024) {
		LOGF("PSSR (1024 requests): render heights %u to %u, %.0f%% of the output's pixels on "
		     "average, output %ux%u, %u by DLSS; GPU thread time a request: %.2f ms before the "
		     "handle, %.2f ms taking it, %.2f ms recording the upscale\n",
		     min_height, max_height,
		     100.0 * static_cast<double>(pixels) /
		         (1024.0 * static_cast<double>(output_width) * output_height),
		     output_width, output_height, dlss_count,
		     static_cast<double>(g_pssr_ns[0]) / 1024e6, static_cast<double>(g_pssr_ns[1]) / 1024e6,
		     static_cast<double>(g_pssr_ns[2]) / 1024e6);
		g_pssr_ns[0] = g_pssr_ns[1] = g_pssr_ns[2] = 0;
		count = dlss_count = 0;
		min_height         = UINT32_MAX;
		max_height         = 0;
		pixels             = 0;
	}
	static uint32_t last[6] {};
	static uint32_t lines = 0;
	const uint32_t  now[6] {request.width, request.height, output_width, output_height,
	                       request.flags, dlss ? 1u : 0u};
	if (std::equal(std::begin(now), std::end(now), std::begin(last)) || lines >= 64) {
		return;
	}
	std::copy(std::begin(now), std::end(now), std::begin(last));
	lines++;
	LOGF("PSSR: the game renders %ux%u, upscaled to %ux%u by %s (flags 0x%x)\n", request.width,
	     request.height, output_width, output_height, dlss ? "DLSS" : "linear scaling",
	     request.flags);
}

} // namespace

void RenderExecutor::DestroyPssr() {
	for (const auto& context: m_dlss_contexts) {
		if (context.feature != nullptr) {
			Dlss::ReleaseFeature(context.feature);
		}
	}
	const auto device = m_context.GetGraphics().device;
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

bool RenderExecutor::Upscale(vk::CommandBuffer command, const Pssr::Request& request,
                             Dlss::Inputs& inputs) {
	const auto release = [&](Dlss::Feature* feature) {
		if (feature != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [feature] { Dlss::ReleaseFeature(feature); });
		}
	};
	auto context = std::ranges::find(m_dlss_contexts, request.context, &DlssContext::context);
	bool reset   = (request.flags & Pssr::kReset) != 0;
	if (context == m_dlss_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size no quality mode takes scales linearly.
		const auto creation = Dlss::Choose(*Dlss::Capabilities(), request, inputs);
		if (!creation) {
			return false;
		}
		if (context != m_dlss_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_dlss_contexts.size() == 4) {
				release(m_dlss_contexts.front().feature);
				m_dlss_contexts.erase(m_dlss_contexts.begin());
			}
			context = m_dlss_contexts.insert(m_dlss_contexts.end(), {request.context, {}, nullptr});
		}
		context->creation = *creation;
		context->feature  = Dlss::CreateFeature(command, *creation);
		reset             = true;
	}
	if (context->feature == nullptr) {
		return false;
	}
	MemoryBarrier(command);
	if (!Dlss::Evaluate(command, context->feature, request, inputs, reset)) {
		release(context->feature);
		context->feature = nullptr;
		return false;
	}
	MemoryBarrier(command);
	return true;
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	EXIT_IF(buffer.IsInvalid());
	const auto pssr_start = PssrNow();
	m_context.GetCommandScheduler().PopPendingOperations(false);
	const RendererLock lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS, which writes the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	if (Dlss::Capabilities() != nullptr && Dlss::Accepts(request) &&
	    parameters.output_channels[0] == 0 && parameters.output_channels[1] == 1 &&
	    parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	buffer.EndRendering();
	const auto pssr_before_handle = PssrNow();
	const auto command            = buffer.Handle();
	const auto pssr_after_handle  = PssrNow();
	g_pssr_ns[0] += pssr_before_handle - pssr_start;
	g_pssr_ns[1] += pssr_after_handle - pssr_before_handle;
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (dlss_bindings[0]) {
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto& full   = binding.desc.info.extent;
			const vk::Extent2D extent {(std::max)(1u, full.width >> view.base_level),
			                          (std::max)(1u, full.height >> view.base_level)};
			// DLSS takes a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
				               view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		const bool upscaled = Upscale(command, request, inputs);
		g_pssr_ns[2] += PssrNow() - pssr_after_handle;
		NotePssr(request, parameters.output_size[0], parameters.output_size[1], upscaled);
		if (upscaled) {
			ResetBindings();
			return;
		}
	}
	if (!dlss_bindings[0]) {
		NotePssr(request, parameters.output_size[0], parameters.output_size[1], false);
	}
	auto& graphics = m_context.GetGraphics();
	if (m_pssr_pipeline == nullptr) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = bindings.size();
		descriptor_info.pBindings = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
		                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
		                     "create upscaling descriptor layout");
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
		                                  sizeof(PssrParameters)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount = 1;
		layout_info.pSetLayouts = &m_pssr_descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(
		                         &layout_info, nullptr, &m_pssr_pipeline_layout),
		                     "create upscaling pipeline layout");
		const auto shader = CompileSPV(GPU_PSSR_LINEAR_SPV, graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.stage.stage = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = shader;
		create.stage.pName = "main";
		create.layout = m_pssr_pipeline_layout;
		const auto result = graphics.device.createComputePipelines(
		    nullptr, 1, &create, nullptr, &m_pssr_pipeline);
		graphics.device.destroyShaderModule(shader, nullptr);
		RequireVulkanSuccess(result, "create upscaling pipeline");
	}
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, input.image_view, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, output.image_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
		                                 : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout,
	                             0, writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                      0, sizeof(parameters), &parameters);
	command.dispatch((parameters.output_size[0] + 7) / 8,
	                 (parameters.output_size[1] + 7) / 8, 1);
	ShaderAccessBarrier(command, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
