#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTPAGESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTPAGESTATS_H_

#include <cstdint>

// Diagnostics (KYTY_HOST_PAGE_STATS=1, 2026-10-24): which guest pages (the buffer cache's 16 KB
// pages) the walks for shaders that read memory through addresses upload again and again, and
// what else uses those pages, to choose the pages whose GPU copies could be the game's memory
// itself (imported, VK_EXT_external_memory_host) instead of uploads. Every 10 s a "Host pages:"
// report: the walks' uploads, the pages uploaded in most frames ("hot"), how many of those
// buffer bindings also read or upload, the GPU writes, and the largest hot ranges. GPU thread.
namespace Libs::Graphics::HostPageStats {

[[nodiscard]] bool Enabled();
// A walk's upload of [vaddr, vaddr + size).
void NoteWalkUpload(uint64_t vaddr, uint64_t size);
// A buffer binding over [vaddr, vaddr + size), and whether its sync uploaded bytes of it.
void NoteBinding(uint64_t vaddr, uint64_t size, bool written, bool uploaded);
// The GPU's writes to [vaddr, vaddr + size) through a binding.
void NoteGpuWrite(uint64_t vaddr, uint64_t size);
// At each guest frame's end (the suspend point).
void NoteFrame();

} // namespace Libs::Graphics::HostPageStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTPAGESTATS_H_
