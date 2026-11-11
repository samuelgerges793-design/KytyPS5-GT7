#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VKPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VKPROBE_H_

namespace Libs::Graphics {

// Round 21 diagnosis (KYTY_VK_PROBE=1): see vkProbe.cpp.
void VkProbe(const char* stage);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VKPROBE_H_
