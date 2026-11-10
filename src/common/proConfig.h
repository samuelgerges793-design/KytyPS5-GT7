#ifndef EMULATOR_SRC_COMMON_PROCONFIG_H_
#define EMULATOR_SRC_COMMON_PROCONFIG_H_

#include <cstdint>
#include <filesystem>

// Round 21: the PS5 Pro mode options (--trinity, --dlss, --dlss-preset). They are kept out of
// emulatorConfig.h, which the shader recompiler includes: a change to that header changes the
// shader disk cache's source version and every stored shader translation is made again.
namespace Config {

struct ProOptions {
	bool enabled = false;
	// Folder with the user's DLSS model; DLSS upscales PSSR in Pro mode when set.
	std::filesystem::path dlss_folder;
	uint32_t              dlss_preset = 0; // 0 is the model default
};

inline ProOptions g_pro_options;

inline bool Ps5ProEnabled() {
	return g_pro_options.enabled;
}
inline const std::filesystem::path& GetDlssFolder() {
	return g_pro_options.dlss_folder;
}
inline uint32_t GetDlssPreset() {
	return g_pro_options.dlss_preset;
}

} // namespace Config

#endif // EMULATOR_SRC_COMMON_PROCONFIG_H_
