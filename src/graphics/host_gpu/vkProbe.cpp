// Round 21 diagnosis (KYTY_VK_PROBE=1): creates and destroys a Vulkan device on the NVIDIA GPU
// with the extensions DLSS asks for, at stages of the emulator's start, to find which stage leaves
// the process in a state the driver refuses (vkCreateDevice: ErrorInitializationFailed).
#include "graphics/host_gpu/vkProbe.h"

#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <windows.h>

#include <cstdio>
#include <cstdlib>

#define KYTY_PROBE_LOAD(name) auto name = reinterpret_cast<PFN_##name>(GetProcAddress(lib, #name))

namespace Libs::Graphics {

// What the process holds in the range the driver's CUDA part reserves for a device with
// VK_NVX_binary_import (0x200000000 to 0x1000000000, measured in a process of its own).
static void Occupancy(const char* stage, const char* when) {
	MEMORY_BASIC_INFORMATION mbi;
	unsigned long long       addr  = 0x200000000ull;
	unsigned long long       held  = 0;
	unsigned long long       start = 0;
	unsigned long long       end   = 0;
	int                      runs  = 0;
	const auto               flush = [&] {
		if (end != 0 && runs++ < 24) {
			::printf("VkProbe %s (%s):   held 0x%012llx .. 0x%012llx  %.2f GB\n", stage, when, start,
			         end, static_cast<double>(end - start) / 1073741824.0);
		}
		start = end = 0;
	};
	while (addr < 0x1000000000ull &&
	       VirtualQuery(reinterpret_cast<void*>(addr), &mbi, sizeof(mbi)) != 0) {
		const auto base = reinterpret_cast<unsigned long long>(mbi.BaseAddress);
		if (mbi.State != MEM_FREE) {
			held += mbi.RegionSize;
			if (end != base) {
				flush();
				start = base;
			}
			end = base + mbi.RegionSize;
		}
		addr = base + mbi.RegionSize;
	}
	flush();
	::printf("VkProbe %s (%s): %.2f GB held in 0x200000000..0x1000000000, %d runs\n", stage, when,
	         static_cast<double>(held) / 1073741824.0, runs);
	::fflush(stdout);
}

void VkProbe(const char* stage) {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_VK_PROBE");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	if (mode == 0) {
		return;
	}
	Occupancy(stage, "now");
	if (mode != 1) {
		return;
	}
	HMODULE lib = LoadLibraryA("vulkan-1.dll");
	if (lib == nullptr) {
		return;
	}
	KYTY_PROBE_LOAD(vkCreateInstance);
	KYTY_PROBE_LOAD(vkDestroyInstance);
	KYTY_PROBE_LOAD(vkEnumeratePhysicalDevices);
	KYTY_PROBE_LOAD(vkGetPhysicalDeviceProperties);
	KYTY_PROBE_LOAD(vkCreateDevice);
	KYTY_PROBE_LOAD(vkDestroyDevice);

	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	VkInstance instance  = nullptr;
	if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
		::printf("VkProbe %s: no instance\n", stage);
		return;
	}
	uint32_t         count = 8;
	VkPhysicalDevice devices[8];
	vkEnumeratePhysicalDevices(instance, &count, devices);
	VkPhysicalDevice gpu = nullptr;
	for (uint32_t i = 0; i < count; i++) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(devices[i], &props);
		if (props.vendorID == 0x10de) {
			gpu = devices[i];
		}
	}
	if (gpu != nullptr) {
		const float             priority = 1.0f;
		VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qci.queueFamilyIndex = 0;
		qci.queueCount       = 1;
		qci.pQueuePriorities = &priority;
		VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		f12.bufferDeviceAddress = VK_TRUE;
		const char* const exts[] {"VK_NVX_binary_import", "VK_NVX_image_view_handle",
		                          "VK_KHR_buffer_device_address"};
		VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.pNext                   = &f12;
		dci.queueCreateInfoCount    = 1;
		dci.pQueueCreateInfos       = &qci;
		dci.enabledExtensionCount   = 3;
		dci.ppEnabledExtensionNames = exts;
		VkDevice       device       = nullptr;
		const VkResult result       = vkCreateDevice(gpu, &dci, nullptr, &device);
		::printf("VkProbe %s: vkCreateDevice with DLSS's extensions: %d %s\n", stage, result,
		         result == VK_SUCCESS ? "OK" : "FAILED");
		::fflush(stdout);
		Occupancy(stage, "with the device");
		if (device != nullptr) {
			vkDestroyDevice(device, nullptr);
		}
		Occupancy(stage, "device destroyed");
	}
	vkDestroyInstance(instance, nullptr);
}

} // namespace Libs::Graphics

#else

namespace Libs::Graphics {
void VkProbe(const char*) {}
} // namespace Libs::Graphics

#endif
