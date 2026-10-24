# Host pages: the GPU using the game's memory where it lies (plan, 2026-10-24)

## Why

These measurements are from an RTX 4070 Ti SUPER with a Ryzen 7 7700X, GT7 at Lago Maggiore:

- **Round 6** ran the sensitivity test. The game's waits for GPU-written bytes are on the critical
  path: about 120 a second, about 5 ms each, behind about 10 command buffers. GPU-thread speed is
  also on the critical path. Fault handling is not.
- **The probe** (KYTY_HOST_IMPORT_PROBE=1) found that VK_EXT_external_memory_host works with
  4 KB alignment, both on the probe's own memory and on game memory through the backing alias.
  The GPU reads it at 22-24 GB/s (its own memory: 225 GB/s) and writes it at 23.6 GB/s. The CPU
  sees the GPU's writes.
- **KYTY_LABELS_AT_COMPLETION=1** showed no fps loss in a Tokyo time trial (24-32 fps).
- **KYTY_HOST_PAGE_STATS=1** (hostPageStats.cpp, the round 12 runs):
  - The walks for address-reading shaders uploaded 36-37 MB a frame over only about 2,700 pages
    of 16 KB (about 44 MB).
  - About 75-80% of those bytes are on pages that are also under buffer bindings and also
    GPU-written through them. That is a CPU<->GPU ping-pong.
  - Only about 20% of the bytes are on pages that nothing else touches.
  - The largest runs are 32-320 KB each, for example 0x1001a00000-0x1003300000 and
    0xe415..-0xe417.., 0xec06f98000, 0x125c3c0000.

## The design

The unit is a "host page": a 16 KB buffer-cache page whose GPU copy is the game's memory itself,
imported. Both access paths use it, and nothing tracks it.

1. **Imports.**
   - Chunks of guest memory (2 MB, aligned, inside one guest mapping) are imported through the
     backing alias pointer (GuestAddressSpace FindBacking; see hostImportProbe.cpp for the import
     calls, eHostAllocationEXT).
   - Each chunk gets a VkBuffer with AllFlags | eShaderDeviceAddress.
   - The chunks are released when the guest unmaps them. That needs a hook where the buffer cache
     handles unmaps.
2. **Promotion** (GPU thread, from the stats machinery made lightweight):
   - What is promoted: runs of pages that walks uploaded in 40% or more of the frames over the
     last 10 s.
   - When: only at a moment when the run is neither GPU-modified nor CPU-dirty in the memory
     tracker. Otherwise try again later.
   - Steps: UntrackMemory for the run (no protections, faults or dirty state). Point the BDA page
     table entries at the chunk's device address (WriteDataBuffer, as ChangeRegister does at
     bufferCache.cpp ~145). Set a host-page bitmap.
   - Exclusions: runs under texture-cache images, and runs under bindings of 4 MB or more (none
     were seen).
3. **Bindings** (ObtainBuffer / SynchronizeBuffer):
   - A range entirely in host pages of one chunk binds the imported buffer at its offset. It gets
     no sync, no MarkGpuWritten and no stream-buffer copy.
   - A range that is only partly host demotes those pages first:
     - TrackMemory them again.
     - Mark them CPU-dirty, so they upload from guest memory, which holds the latest CPU and GPU
       bytes.
     - Point their BDA entries back at the device buffer.
4. **ChangeRegister<insert> / Unregister** write and clear BDA entries for whole buffers. After
   that, the host pages in the range must get their chunk addresses again.
5. **Readbacks, fast readbacks, readahead and write bitmaps** all work from GPU-modified ranges.
   Host pages are never marked, so nothing downloads over them.
6. **Labels** must be at completion: run with KYTY_LABELS_AT_COMPLETION=1, or make it the
   default when host pages are on. The game then cannot reuse memory the GPU has not read.
7. **Switch and logging.**
   - KYTY_HOST_PAGES=1 turns it on (off by default). Compare whole runs; an A/B would promote and
     demote every 20 s.
   - A log line every 10 s: host pages, chunks, promotions, demotions, redirected bindings.

## Risks

- **Wrong bytes.** Games that pace buffer reuse by frame count, not by labels, could be hit;
  expect glitches if so.
- **Images over host pages.** Exclude them.
- **Guest unmaps.**
- **PCIe bandwidth.** These pages are about 44 MB and read a few times a frame, so about 2 ms of
  bus time a frame. They already cross the bus once now, as uploads.

## Expected gain

- **GPU thread:** most of the buffer-sync and protection work for draws and dispatches, about
  5-6 ms of about 32 ms a frame.
- **Game threads:** the ping-pong's copy-backs and faults.
- **GPU:** the walk uploads and their barriers.
