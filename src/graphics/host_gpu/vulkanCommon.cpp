#include "graphics/host_gpu/vulkanCommon.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {

struct FormatMapping {
	Prospero::BufferFormat guest;
	vk::Format             host;
};

constexpr FormatMapping kFormatMappings[] = {
    {Prospero::BufferFormat::k8UNorm, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8SNorm, vk::Format::eR8Snorm},
    {Prospero::BufferFormat::k8UScaled, vk::Format::eR8Uscaled},
    {Prospero::BufferFormat::k8SScaled, vk::Format::eR8Sscaled},
    {Prospero::BufferFormat::k8UInt, vk::Format::eR8Uint},
    {Prospero::BufferFormat::k8SInt, vk::Format::eR8Sint},
    {Prospero::BufferFormat::k16UNorm, vk::Format::eR16Unorm},
    {Prospero::BufferFormat::k16SNorm, vk::Format::eR16Snorm},
    {Prospero::BufferFormat::k16UScaled, vk::Format::eR16Uscaled},
    {Prospero::BufferFormat::k16SScaled, vk::Format::eR16Sscaled},
    {Prospero::BufferFormat::k16UInt, vk::Format::eR16Uint},
    {Prospero::BufferFormat::k16SInt, vk::Format::eR16Sint},
    {Prospero::BufferFormat::k16Float, vk::Format::eR16Sfloat},
    {Prospero::BufferFormat::k8_8UNorm, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8SNorm, vk::Format::eR8G8Snorm},
    {Prospero::BufferFormat::k8_8UScaled, vk::Format::eR8G8Uscaled},
    {Prospero::BufferFormat::k8_8SScaled, vk::Format::eR8G8Sscaled},
    {Prospero::BufferFormat::k8_8UInt, vk::Format::eR8G8Uint},
    {Prospero::BufferFormat::k8_8SInt, vk::Format::eR8G8Sint},
    {Prospero::BufferFormat::k32UInt, vk::Format::eR32Uint},
    {Prospero::BufferFormat::k32SInt, vk::Format::eR32Sint},
    {Prospero::BufferFormat::k32Float, vk::Format::eR32Sfloat},
    {Prospero::BufferFormat::k16_16UNorm, vk::Format::eR16G16Unorm},
    {Prospero::BufferFormat::k16_16SNorm, vk::Format::eR16G16Snorm},
    {Prospero::BufferFormat::k16_16UScaled, vk::Format::eR16G16Uscaled},
    {Prospero::BufferFormat::k16_16SScaled, vk::Format::eR16G16Sscaled},
    {Prospero::BufferFormat::k16_16UInt, vk::Format::eR16G16Uint},
    {Prospero::BufferFormat::k16_16SInt, vk::Format::eR16G16Sint},
    {Prospero::BufferFormat::k16_16Float, vk::Format::eR16G16Sfloat},
    {Prospero::BufferFormat::k11_11_10Float, vk::Format::eB10G11R11UfloatPack32},
    // 2_10_10_10 keeps its 2-bit component at the bottom, which no Vulkan format does: the host
    // image holds it in alpha (ResolveHostFormat in textureCommon.cpp maps the components).
    {Prospero::BufferFormat::k2_10_10_10UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k2_10_10_10UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k10_10_10_2UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k10_10_10_2SNorm, vk::Format::eA2B10G10R10SnormPack32},
    {Prospero::BufferFormat::k10_10_10_2UScaled, vk::Format::eA2B10G10R10UscaledPack32},
    {Prospero::BufferFormat::k10_10_10_2UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k8_8_8_8UNorm, vk::Format::eR8G8B8A8Unorm},
    {Prospero::BufferFormat::k8_8_8_8SNorm, vk::Format::eR8G8B8A8Snorm},
    {Prospero::BufferFormat::k8_8_8_8UScaled, vk::Format::eR8G8B8A8Uscaled},
    {Prospero::BufferFormat::k8_8_8_8SScaled, vk::Format::eR8G8B8A8Sscaled},
    {Prospero::BufferFormat::k8_8_8_8UInt, vk::Format::eR8G8B8A8Uint},
    {Prospero::BufferFormat::k8_8_8_8SInt, vk::Format::eR8G8B8A8Sint},
    {Prospero::BufferFormat::k32_32UInt, vk::Format::eR32G32Uint},
    {Prospero::BufferFormat::k32_32SInt, vk::Format::eR32G32Sint},
    {Prospero::BufferFormat::k32_32Float, vk::Format::eR32G32Sfloat},
    {Prospero::BufferFormat::k16_16_16_16UNorm, vk::Format::eR16G16B16A16Unorm},
    {Prospero::BufferFormat::k16_16_16_16SNorm, vk::Format::eR16G16B16A16Snorm},
    {Prospero::BufferFormat::k16_16_16_16UScaled, vk::Format::eR16G16B16A16Uscaled},
    {Prospero::BufferFormat::k16_16_16_16SScaled, vk::Format::eR16G16B16A16Sscaled},
    {Prospero::BufferFormat::k16_16_16_16UInt, vk::Format::eR16G16B16A16Uint},
    {Prospero::BufferFormat::k16_16_16_16SInt, vk::Format::eR16G16B16A16Sint},
    {Prospero::BufferFormat::k16_16_16_16Float, vk::Format::eR16G16B16A16Sfloat},
    {Prospero::BufferFormat::k32_32_32UInt, vk::Format::eR32G32B32Uint},
    {Prospero::BufferFormat::k32_32_32SInt, vk::Format::eR32G32B32Sint},
    {Prospero::BufferFormat::k32_32_32Float, vk::Format::eR32G32B32Sfloat},
    {Prospero::BufferFormat::k32_32_32_32UInt, vk::Format::eR32G32B32A32Uint},
    {Prospero::BufferFormat::k32_32_32_32SInt, vk::Format::eR32G32B32A32Sint},
    {Prospero::BufferFormat::k32_32_32_32Float, vk::Format::eR32G32B32A32Sfloat},
    // Narrow-channel sRGB formats are optional in Vulkan. Keep a same-width fallback until
    // sampler-aware sRGB emulation is available.
    {Prospero::BufferFormat::k8Srgb, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8_8Srgb, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8_8_8Srgb, vk::Format::eR8G8B8A8Srgb},
    {Prospero::BufferFormat::k9_9_9_5Float, vk::Format::eE5B9G9R9UfloatPack32},
    {Prospero::BufferFormat::k5_6_5UNorm, vk::Format::eB5G6R5UnormPack16},
    {Prospero::BufferFormat::k5_5_5_1UNorm, vk::Format::eA1R5G5B5UnormPack16},
    {Prospero::BufferFormat::k1_5_5_5UNorm, vk::Format::eR5G5B5A1UnormPack16},
    {Prospero::BufferFormat::k4_4_4_4UNorm, vk::Format::eR4G4B4A4UnormPack16},
    {Prospero::BufferFormat::kBc1UNorm, vk::Format::eBc1RgbaUnormBlock},
    {Prospero::BufferFormat::kBc1Srgb, vk::Format::eBc1RgbaSrgbBlock},
    {Prospero::BufferFormat::kBc2UNorm, vk::Format::eBc2UnormBlock},
    {Prospero::BufferFormat::kBc2Srgb, vk::Format::eBc2SrgbBlock},
    {Prospero::BufferFormat::kBc3UNorm, vk::Format::eBc3UnormBlock},
    {Prospero::BufferFormat::kBc3Srgb, vk::Format::eBc3SrgbBlock},
    {Prospero::BufferFormat::kBc4UNorm, vk::Format::eBc4UnormBlock},
    {Prospero::BufferFormat::kBc4SNorm, vk::Format::eBc4SnormBlock},
    {Prospero::BufferFormat::kBc5UNorm, vk::Format::eBc5UnormBlock},
    {Prospero::BufferFormat::kBc5SNorm, vk::Format::eBc5SnormBlock},
    {Prospero::BufferFormat::kBc6UFloat, vk::Format::eBc6HUfloatBlock},
    {Prospero::BufferFormat::kBc6SFloat, vk::Format::eBc6HSfloatBlock},
    {Prospero::BufferFormat::kBc7UNorm, vk::Format::eBc7UnormBlock},
    {Prospero::BufferFormat::kBc7Srgb, vk::Format::eBc7SrgbBlock},
};

constexpr auto MakeFormatLookup() {
	constexpr auto kMaxFormat = static_cast<size_t>(Prospero::BufferFormat::kBc7Srgb);
	std::array<vk::Format, kMaxFormat + 1> lookup {};
	lookup.fill(vk::Format::eUndefined);
	for (const auto& mapping: kFormatMappings) {
		lookup[static_cast<size_t>(mapping.guest)] = mapping.host;
	}
	return lookup;
}

constexpr auto kFormatLookup = MakeFormatLookup();

} // namespace

vk::Format VulkanFormat(Prospero::BufferFormat guest_format) {
	const auto index = static_cast<size_t>(guest_format);
	return index < kFormatLookup.size() ? kFormatLookup[index] : vk::Format::eUndefined;
}

void RequireVulkanSuccess(vk::Result result, const char* operation) {
	if (result != vk::Result::eSuccess) {
		EXIT("%s failed: %s (%d)\n", operation, vk::to_string(result).c_str(),
		     static_cast<int>(result));
	}
}

vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device,
                            const vk::AllocationCallbacks* allocator) {
	vk::ShaderModuleCreateInfo create_info {};
	create_info.codeSize    = code.size_bytes();
	create_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(device.createShaderModule(&create_info, allocator, &module),
	                     "create SPIR-V shader module");
	return module;
}

namespace {

std::array<std::atomic<int64_t>, static_cast<size_t>(DriverMemoryKind::Count)> g_driver_bytes {};

// Each block keeps its size and the offset of its start in the two words before what the driver
// gets, so frees and reallocations know both.
struct BlockHeader {
	size_t size;
	size_t offset;
};

std::atomic<int64_t>& DriverBytes(void* user_data) {
	return g_driver_bytes[reinterpret_cast<uintptr_t>(user_data)];
}

void* VKAPI_CALL DriverAllocate(void* user_data, size_t size, size_t alignment,
                                vk::SystemAllocationScope /*scope*/) {
	alignment            = std::max(alignment, alignof(std::max_align_t));
	const size_t offset  = Common::AlignUp(sizeof(BlockHeader), alignment);
	const size_t total   = Common::AlignUp(offset + size, alignment);
#if defined(_WIN32)
	auto* const  block   = static_cast<uint8_t*>(_aligned_malloc(total, alignment));
#else
	auto* const  block   = static_cast<uint8_t*>(std::aligned_alloc(alignment, total));
#endif
	if (block == nullptr) {
		return nullptr;
	}
	auto* const memory = block + offset;
	const BlockHeader header {size, offset};
	std::memcpy(memory - sizeof(BlockHeader), &header, sizeof(BlockHeader));
	DriverBytes(user_data).fetch_add(static_cast<int64_t>(size), std::memory_order_relaxed);
	return memory;
}

void VKAPI_CALL DriverFree(void* user_data, void* memory) {
	if (memory == nullptr) {
		return;
	}
	BlockHeader header {};
	std::memcpy(&header, static_cast<uint8_t*>(memory) - sizeof(BlockHeader), sizeof(BlockHeader));
	DriverBytes(user_data).fetch_sub(static_cast<int64_t>(header.size), std::memory_order_relaxed);
#if defined(_WIN32)
	_aligned_free(static_cast<uint8_t*>(memory) - header.offset);
#else
	std::free(static_cast<uint8_t*>(memory) - header.offset);
#endif
}

void* VKAPI_CALL DriverReallocate(void* user_data, void* original, size_t size, size_t alignment,
                                  vk::SystemAllocationScope scope) {
	if (original == nullptr) {
		return DriverAllocate(user_data, size, alignment, scope);
	}
	if (size == 0) {
		DriverFree(user_data, original);
		return nullptr;
	}
	BlockHeader header {};
	std::memcpy(&header, static_cast<uint8_t*>(original) - sizeof(BlockHeader),
	            sizeof(BlockHeader));
	void* const memory = DriverAllocate(user_data, size, alignment, scope);
	if (memory == nullptr) {
		return nullptr;
	}
	std::memcpy(memory, original, std::min(size, header.size));
	DriverFree(user_data, original);
	return memory;
}

bool DriverMemoryCounted() {
	static const bool counted = [] {
		const char* value = std::getenv("KYTY_VK_HOST_MEMORY");
		return value == nullptr || value[0] != '0';
	}();
	return counted;
}

} // namespace

const vk::AllocationCallbacks* DriverAllocator(DriverMemoryKind kind) {
	static const auto callbacks = [] {
		std::array<vk::AllocationCallbacks, static_cast<size_t>(DriverMemoryKind::Count)> result {};
		for (size_t i = 0; i < result.size(); i++) {
			result[i].pUserData       = reinterpret_cast<void*>(i);
			result[i].pfnAllocation   = &DriverAllocate;
			result[i].pfnReallocation = &DriverReallocate;
			result[i].pfnFree         = &DriverFree;
		}
		return result;
	}();
	return DriverMemoryCounted() ? &callbacks[static_cast<size_t>(kind)] : nullptr;
}

uint64_t DriverMemoryBytes(DriverMemoryKind kind) {
	return static_cast<uint64_t>(
	    std::max<int64_t>(0, g_driver_bytes[static_cast<size_t>(kind)].load(std::memory_order_relaxed)));
}

} // namespace Libs::Graphics
