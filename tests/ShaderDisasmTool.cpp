// Disassembles RDNA2 shader binaries dumped with --graphics-debug-dump (the files under
// <shader-log-folder>/original) and the SPIR-V the recompiler produced for them (*.spv).
// Usage: shader_disasm FILE...
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	using namespace Libs::Graphics::ShaderRecompiler;
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s FILE...\n", argv[0]);
		return 1;
	}
	for (int i = 1; i < argc; i++) {
		std::ifstream     file(argv[i], std::ios::binary);
		std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
		if (!file && !file.eof()) {
			std::fprintf(stderr, "%s: cannot read\n", argv[i]);
			return 1;
		}
		std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
		std::memcpy(code.data(), bytes.data(), code.size() * sizeof(uint32_t));
		const std::string name(argv[i]);
		if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spv") == 0) {
			spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
			std::string         text;
			if (!tools.Disassemble(code, &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) {
				std::fprintf(stderr, "%s: invalid SPIR-V\n", argv[i]);
				return 1;
			}
			std::printf("=== %s (%zu words)\n%s\n", argv[i], code.size(), text.c_str());
			continue;
		}
		Decoder::Program program;
		Decoder::DecodeProgram(code, program);
		std::printf("=== %s (%zu words)\n%s\n", argv[i], code.size(),
		            Decoder::ProgramToString(program).c_str());
	}
	return 0;
}
