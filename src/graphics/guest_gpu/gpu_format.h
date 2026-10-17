#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_

#include "graphics/guest_gpu/gpu_defs.h"

namespace Libs::Graphics::Prospero {

enum class ChannelOrderSupport : uint8_t {
	kNone,
	kStandardOnly,
	kAll,
};

struct RenderTargetFormatEncoding {
	BufferFormat        buffer_format = BufferFormat::kInvalid;
	uint8_t             components    = 0;
	ChannelOrderSupport order_support = ChannelOrderSupport::kNone;

	[[nodiscard]] constexpr bool IsValid() const {
		return buffer_format != BufferFormat::kInvalid && components >= 1u && components <= 4u &&
		       order_support != ChannelOrderSupport::kNone;
	}

	[[nodiscard]] constexpr bool SupportsOrder(ChannelOrder order) const {
		switch (order_support) {
			case ChannelOrderSupport::kStandardOnly: return order == ChannelOrder::kStandard;
			case ChannelOrderSupport::kAll:
				switch (order) {
					case ChannelOrder::kStandard:
					case ChannelOrder::kAlt:
					case ChannelOrder::kReversed:
					case ChannelOrder::kAltReversed: return true;
				}
				return false;
			default: return false;
		}
	}
};

enum class TextureNumericClass { Unsupported, Float, Uint, Sint };

RenderTargetFormatEncoding ResolveRenderTargetFormat(ChannelLayout layout, ChannelType type);
uint32_t                   NumBytesPerElement(BufferFormat format);
uint32_t                   BlockCompressedBytesPerBlock(BufferFormat format);
uint32_t                   RenderTargetBytesPerElement(BufferFormat format);
bool                       IsFmaskTextureFormat(BufferFormat format);
TextureNumericClass        SampledTextureNumericClass(BufferFormat format);
BufferFormat               RemapTextureFormat(BufferFormat format);
// UNorm render-target formats stored as raw 32-bit words, packed by the pixel shader.
[[nodiscard]] constexpr bool IsPackedUnormTargetFormat(BufferFormat format) {
	return format == BufferFormat::k11_11_10UNorm || format == BufferFormat::k2_10_10_10UNorm;
}

} // namespace Libs::Graphics::Prospero

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_ */
