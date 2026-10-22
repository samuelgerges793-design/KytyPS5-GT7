# Cloud session notes (2026-10-22)

These notes are from the cloud session, which had a Linux container with no GPU and no game. The
local Windows session can read them and continue from here. Each item says what was measured and
what is an estimate.

## Branches (samuelgerges793-design/KytyPS5-GT7)

| Branch | Content | PR |
| --- | --- | --- |
| `claude/gt7-base` | **The proposed new base for `gt7-local`; it fast-forwards from it.** It is `gt7-local` plus #1, #2 and #3, plus this file. | #5 |
| `claude/pad-file-input` | `KYTY_PAD_FILE` scripted input, in the old branch's format, with a safe stop at shutdown | #1 |
| `claude/pipeline-cache-fixes` | Must-wait cap per pipeline; snapshots with headroom and retries; atomic file replace (`File::RenameReplacing`); env vars clamped | #2 |
| `claude/upstream-b46eeb7` | 29 upstream picks (22 safe, plus a 7-commit shader bundle); two test fixes | #3 |
| `claude/gt7-test` | The build tested on Windows (`906bae38`). `claude/gt7-base` replaces it. | #4 |

Tested on Windows (RTX 4070 Ti Super), build `906bae38`:
- **Game:** a 10-lap Northern Isle race, unattended through the pad file, ran with no crash. It averaged 16 fps (14-18); `gt7-local` gave 15-20. The pipeline-cache snapshots wrote at 671, 767 and 878 MB.
- **ctest:** 7 of 54 tests failed. Six of them fail identically on `gt7-local` (listed under Known problems).
- **The 7th, `shader_recompiler_compute`,** segfaulted in `CheckComparisonDepthTexture`. `gt7-base` drops upstream's `ImageGatherCompareLod32/64` cases (commit `ad092f5f`), so that function is identical to `gt7-local`'s again.

**The shader bundle** is the 7 commits from `7d933a22` "shader: decide scalar branches by a vote over the whole guest wave" through `9d5a0134`. If it misbehaves, revert those commits; they make most shaders compile once more.

## How the emulator spends a race frame

- **Threads:**
  - `Thread_Gpu` (`graphicsRun.cpp`, `GuestGpu::ThreadRun`) is a single host thread. It runs every guest queue, graphics and compute, in turn, and prepares every draw: `pm4Handlers.cpp`, `renderDraw.cpp`, `renderCompute.cpp`.
  - Vulkan recording is done by `GpuRecord`, and submission by `GpuQueueSubmit` (`commandScheduler.cpp`).
  - `Thread_DrawPrep` walks the draws ahead of the GPU thread.
  - Pipelines compile in the background on `PipelineCompiler`, `ShaderTranslator` and `ComputePipelineCompile`.
- **Measured by usa-ratt's rounds** (7950X3D + RTX 5090; not re-measured on the 7700X):
  - The GPU thread sets the frame rate.
  - In races, the GPU thread and the game's main job thread are both near their limit (Round 113).
  - The GPU thread spends about 13 % of its time on host page-protection changes and about 12 % copying guest memory for uploads (Round 111).
  - The reflection cube is 550-750 draws a frame, 30-40 % of the GPU thread's draw time. Drawing it every other frame gave only +0.8 fps in a race, but +4 fps in a time trial. In races, the game thread limits as well.
- **Diagnostics, already in `_kyty.txt`:**
  - `Perf` every 2 s; `Perf threads` gives CPU and fault time per thread.
  - `GPU phases` every 10 s gives microseconds per draw step and the per-queue split ("by queue").
  - `KYTY_SAMPLE=1` is a sampling profiler that names functions from the `.pdb` (Windows only).
  - `KYTY_BISECT="a;b;c"` plus F8 switches between A/B steps within one run, and the `Perf` lines name the active step (`renderer/bisect.h`).

## Open leads for race speed (cheapest first, none measured on this PC)

The pad-file harness now makes runs repeatable, so measure each lead against the same race.

1. **Sensitivity first.** Run `KYTY_BISECT="slow:gpu;slow:faults;slow:reads"` (Round 114; or `run-gt7-slowtest.cmd`). Whichever slowdown costs the most fps marks the thread to work on: the GPU thread, the game threads' fault handling, or readback waits.
2. **Build config.** Releases build with LTO (`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`); `build-ratt.cmd` doesn't. Also try `-DKYTY_CPU_ARCH=znver4` (Round 113, never measured). Compare like with like.
3. **`KYTY_ASYNC_MEMORY_SYNC=1`** (or the F8 step `memsync:async`). It moves the ~25 % of protection changes and upload copies off the GPU thread. It's off by default and was never measured in a race, so watch for glitches.
4. **The launcher's "Reflections every other frame"** option (`--reflections-every 2`): +0.8 fps in usa-ratt's race.
5. **Upstream `93452b24` "renderer: cache dynamic graphics state".** It saves about 15 `vkCmdSet*` calls per draw on the GPU thread. The port is moderate to hard: a 203-line conflict, mostly in `renderDraw.cpp`.
6. **Game-thread costs.**
   - `--redzone` routes memory instructions in red-zone functions through trampolines (`loader/guestInstructionPatcher.cpp`), and `KYTY_SAMPLE` shows samples landing on them.
   - Write faults on GPU-tracked pages show in `Perf threads`.
   - Size both with the sampler before changing anything.
7. **A host thread of their own for the compute queues.** The "by queue" line shows the potential. It's a large, risky change, because the caches assume a single thread.

The 120 Hz mode is not a realistic target. Even 25-30 fps in races would need about 1.5x on both threads.

## Pending (not on gt7-base)

- **usa-ratt's release `1e1de7b` (Rounds 115-118).**
  - It fixes the race crash with `resource.mip_count` (their issues 3 and 9) and the ray tracing materials.
  - Every shader compiles again after updating.
  - Merging it into `gt7-local` or `gt7-base` conflicts in one hunk, at the start of `PipelineCache::Save()`. Keep both the snapshot-thread stop/join and their `m_saved = true;`, after checking what `m_saved` guards.
- **Upstream `f5f4e7d4`** "usbd: return an empty device list from sceUsbdGetDeviceList (Gran Turismo 7)".
  - The fork lacks NIDs `8qB9Ar4P5nc` and `EQ6SCLMqzkM`. A call to either lands in the unresolved-import stub (`runtimeLinker.cpp`), which logs `Unresolved import stub called: <symbol>` and returns 0 without writing the list pointer.
  - It cherry-picks cleanly: `git fetch https://github.com/KytyPS5/KytyPS5.git main && git cherry-pick -x f5f4e7d4`.
- **Upstream commits since b46eeb7 that the fork lacks, worth porting only if the logs show the opcodes:** `d7c18966` (`V_INTERP_P1LL_F16`/`P2_F16`), and `753f2851` + `c5016d94` (D16 XY/XYZ buffer formats). Look for `gave up hash` or `unsupported family` lines.
- **From the earlier triage, not taken:**
  - `feed075a`: real user clip planes. GT7 sets them (Tokyo, the LM55 page), and the fork draws those draws unclipped.
  - `70668fbd`: global barriers. Ordering risk.
  - `1ce663b5`: tessellation fixes. More than a day's port.
  - `a6f4e8e7`: video sync, for issue 4.
  - `2d00e13b`: the SrtWalker must change with it.

## Known problems

- **Six ctest failures also fail on `gt7-local`:**
  - `pm4_context_state` ("legacy register holes")
  - `compute_meta_clear_classification` and `texture_cache_image_overlap` ("HTile allocation reused as expanded color metadata")
  - `buffer_cache_dirty_gc` ("adjacent download reservation stride")
  - `texture_cache_htile_clear` and `buffer_cache_ranges` ("DCC alias host layer")

  `shader_recompiler_compute`'s main run also stops at `StorageTextureGpuOwnedRebind` on `gt7-local`.
- **The upstream `ImageGatherCompareLod32/64` cases segfault on NVIDIA but pass on Mesa llvmpipe.** The cause is unknown. To debug, restore them from `6161ce0f` and run `--image-gather-lod-only` under a debugger. The decoder now accepts `IMAGE_GATHER4_C_L`, so watch for problems if GT7 uses it.

## Working notes

- **Linux build:** `cmake -S . -B _Build/linux-no-qt -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKYTY_BUILD_LAUNCHER=OFF`.
- **Vulkan tests without a GPU:** they need fragment barycentrics, 64-bit image atomics and image-view min LOD. Only for diagnosis, the harness's checks for these can be relaxed to run under llvmpipe (`VulkanHarness::Init`).
- **Formatting:** the pre-commit hook pins clang-format 22.1.3 (`pip install clang-format==22.1.3`).
- **`shader_cfg` SPIR-V size budgets are upper bounds.** When a change grows a shader, raise its budget by the measured growth.
- **The root `AGENTS.md` is a stale handoff for another game;** ignore it.
- **Pipeline-cache env vars:**
  - `KYTY_PIPELINE_MUST_WAIT_MS`: default 10 s. The budget is per pipeline: later draws on that pipeline skip until it's ready.
  - `KYTY_PIPELINE_SAVE_MINUTES`: default 3; 0 or less saves only at exit.
