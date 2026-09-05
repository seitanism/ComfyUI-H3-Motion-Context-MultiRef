# Fork history and modifications

This repository is a modified version of **NikoDemon80/ComfyUI-H3-Motion-Context**.

- Original project: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Original author: **NikoDemon80**
- License: **GPL-3.0**
- H3 AV masking design credit: **Barish Ozbay (`drozbay`)**, author of ComfyUI PR #15375

Low-level implementation details live in [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

---


## Update 9 — Audit repairs, exact workflow timing, and compatibility — 2026-09-05

This package follows Update 8 without renumbering Reithan’s Update 7 contribution. It is intended for local ComfyUI testing before publication. CPU/static verification does not certify full H3 inference, browser behavior, or external-pack compatibility.

### Runtime fixes

- Fixed a first-frame tensor-view retention in the direct VHS streamers. The first preview/probe frame now owns only its own storage, allowing the first full decoded clip to be released while later clips are encoded. VHS still owns filename numbering.
- Fixed spatial V2V masks to follow the source’s center crop before latent resizing. Global V2V settings are unchanged.
- Corrected shortened Motion Context timeline-audio placement using actual returned audio coverage. Complete windows, public names, argument ordering, and saved widget ordering retain their existing behavior; coordinates still use the nearest representable video frame.
- Current workflows validate constant-24-fps source/loaded metadata and lock H3/source/output clock controls at 24. Old Python normalization helpers are retained for legacy integrations. The check cannot infer VFR timing from frames/average-FPS metadata; convert source videos to constant 24 fps first.
- AV Bridge contexts now require **39 + 51*k** frames (39, 90, 141, 192, …); 56 is rejected. Full targets still use **5 + 17*k**, with a generated middle longer than zero. The new H3 AV Bridge Timing node validates both controls before generation.
- Replaced the AV Bridge graph’s FPS-dependent trim/math/concatenation chain with H3 Assemble Bridge Audio. It cuts exact protected audio cells, conforms the generated middle to the 24 fps picture interval, and assembles both sources on absolute PCM boundaries. Non-shared target endpoints remain supported without drift.

### Audio preservation clarification

The existing eight-tick half-cosine audio feather is retained. For the default 39-frame context, **57 of 65 audio ticks remain fully protected**; only the last **8 ticks (0.2 s)** gradually receive denoising. Keep source audio does **not** regenerate the entire 1.625-second context. Final AV streaming uses the extension’s decoded overlap to retain that feather. VAE reconstruction and bounded timebase conformance can still make small changes to the decoded overlap; the policy does not promise bit-identical source PCM. The interior-insert assembler has a different purpose: it restores original canonical source pixels/PCM over its entire inserted interval, replacing any feather there.

### PR #15972 — audio input cropping

The supplied Update 8 snapshot used exact-grid PCM preparation, not a global crop monkeypatch. That source-side correction remains active: every repo-owned audio encode constructs its intended exact H3 cell span and checks the encoded length.

Added `h3_audio_vae_compat.py` and H3 Audio VAE Compatibility. On use, an unfixed connected H3 audio VAE gets `crop_input=False`, equivalent to [ComfyUI PR #15972](https://github.com/Comfy-Org/ComfyUI/pull/15972) by seitanism. Current workflow audio VAEs pass through this node before stock reference encodes too. Already-fixed native instances and unrelated VAEs are unchanged; no VAE constructor is globally patched. Exact-grid preparation remains necessary for semantic start/end anchoring after the upstream fix.

The regression reproduces 437,333 samples being cropped to 546 cells with a shifted origin, then verifies an unchanged origin and 547 cells with the correction; native-fix retirement is tested.

### PR #15988 — masked velocity conversion

Included a lazy, H3-only compatibility equivalent of [poorpaper’s ComfyUI PR #15988](https://github.com/Comfy-Org/ComfyUI/pull/15988). Returned video/audio velocities are multiplied by their masks before audio carry conversion and outer global-sigma x0 conversion. This makes the predicted local `mask * sigma` timestep consistent with recovery, including fractional/feathered masks.

The live forward is probed with synthetic outputs, including audio carry scale 4. Native passing behavior is left untouched. The fallback supports both the original wrapper output and the newer malloc-graph output-copy stage; it preserves those operations and inserts scaling before audio carry conversion. Unknown/partially converted code is rejected instead of double-patched, and a failed post-check restores the original forward. The patch is process-wide for H3 after first masked-node use and disappears on restart.

Both PRs were still open when inspected on 2026-09-05. Runtime retirement depends on the **installed behavior**, not an online merge-status check: native/backported fixes win, while an older local build still needs compatibility after an upstream merge.

### Issue #16 — dated output folders

Final AV/Music streamers now accept `%date:yyyy-MM-dd%/MiniMax_H3_`, including API queues without VHS frontend hooks. Supported tokens are `yyyy`, `yy`, `MM`, `dd`, `HH`, `hh`, `mm`, and `ss`; dates use the server’s local clock. Relative subdirectories are supported under ComfyUI output, and VHS retains its numbered filename allocation. Absolute paths and traversal prefixes are rejected. Addresses [issue #16](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/issues/16).

### Precision, documentation, and verification

- Clarified the finite 1/4096 ceiling-quantized compatibility grid in the README, V2V workflow note, and architecture. Near-one values above 4095/4096 enter the 1.0 bin; tiny slider changes may be equivalent. Native implementations may offer finer precision.
- Replaced the audio-shortcut literal-string readiness assumption with an extracted-branch behavior probe. Compatibility rewriting still requires a recognized source/AST shape.
- Added [KEYFRAMES_AND_INSERTS.md](KEYFRAMES_AND_INSERTS.md) for soft/hard anchors, phase grids, mask composition, interior splice recipes, and related Update 7 nodes, with attribution to Reithan. Corrected latent-versus-pixel guarantees and incoming audio-mask semantics.
- Added [WORKFLOW_PREREQUISITES.md](WORKFLOW_PREREQUISITES.md) with each current example’s external packs, recorded versions, exact selected model filenames, and required inputs. Recorded workflow versions are evidence, not new end-to-end compatibility guarantees.
- Kept the casual README compact and moved detailed changes here. Updated the technical architecture’s current/historical distinctions, memory contract, and patch scope.
- Added pytest to CI and `requirements-test.txt`. Each test module now runs in a fresh pytest process, honoring fixtures and exposing previously incomplete mocks; repaired those mocks and updated obsolete topology assertions to check the new contracts.
- Release validation: **214 CPU/static/mock checks passed** with real pytest fixture execution; all Python files parsed, JavaScript syntax passed, and all nine workflow graphs passed reciprocal-link checks. The three OLD workflow JSON files are unchanged. No H3 model/GPU inference or live ComfyUI browser run was available.
- Regenerated SHA256SUMS after all release edits. The checksum helper supports both generation and verification without hashing its own manifest or transient caches.


## Initial fork — MultiRef + Motion Context coexistence

The fork began by adapting H3 Motion Context for workflows that also used MiniMax H3 reference conditioning.

Main goals included preserving normal Ref2VA/MultiRef references while carrying previous visual/audio context into continuation clips, and keeping compatibility behavior inside the custom-node pack rather than permanently modifying ComfyUI files on disk.

---

## Update 1 — Custom keyframes — 2026-08-10

Merged as [PR #1](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/1).

### Main changes

- Added **H3 Custom Keyframes**.
- Added arbitrary visual keyframe positions on the H3 target timeline.
- Added the Custom Keyframes example workflow.
- Added lazy runtime compatibility installation for the functionality required at the time.

---

## Update 2 — Existing-video extension and compatibility improvements — 2026-08-12

Merged as [PR #2](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/2).

### Main changes

- Added **H3 Existing Video Masked Context**.
- Added **H3 Assemble Existing Video Extension**.
- Added **H3 Crop Source To /32**.
- Added the advanced input-video extension workflow with optional Ref2VA/MultiRef images.
- Added configurable continuation context and visual overlap blending.
- Improved music-video song-slice timing.
- Reworked compatibility handling to be capability-aware: use native ComfyUI behavior when available and activate local compatibility only when required.
- Added CPU/static regressions for timing, masking, assembly, and workflow structure.

Update 2 introduced the first practical source-video latent-masking path. Later updates expanded that idea into general per-token AV masking and long multi-clip chains.

---

## Compatibility milestone — native H3 guide architecture — 2026-08-13

This was an important backend migration, but **not a separate numbered update PR**.

As ComfyUI gained native H3 arbitrary guide/reference coexistence, the classic Motion Context path was moved away from the older layout/payload monkey-patching design and toward native H3 keyframe/audio-guide conditioning.

The legacy Motion Context workflows therefore remain guide-based workflows, but they use the newer native H3 mechanisms where available. See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) for details.

---

## Update 3 — Per-token video/audio latent masking — 2026-08-14

Merged as [PR #4](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/4).

### Main changes

- Added per-token noise masking for MiniMax H3 video and audio latents.
- Added **H3 Masked AV Bridge**.
- Updated the existing-video masked extension path for the post-native-guide H3 architecture.
- Added one-video masked extension and two-video masked bridge examples.
- Added exact audio trimming/assembly for masked extension workflows.
- Removed obsolete pre-native-guide layout/payload monkey-patching where it was no longer needed.
- Added additional masking and workflow regressions.

Typical latent-mask semantics are:

- `0` = preserve known content;
- `1` = denoise/generate new content.

This became the foundation for the later latent-masking workflow family.

---

## Update 4 — Exact song-latent masking — 2026-08-14

Merged as [PR #6](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/6).

### Main changes

- Added **H3 Song Audio + Masked Video Context** (`MiniMaxH3SongMaskedAVContext`).
- Added the original master song directly to the target H3 audio latent.
- Protected the song audio latent from denoising.
- Allowed previous visual context to be protected independently while future video remained generative.
- Kept normal Ref2VA image conditioning available without using the song as a `ref_audio_*` reference.
- Added the FL2VA/master-song music-video example and reproducible media assets/tests.
- Used the untouched master song as the authoritative final soundtrack.

Update 4 established the exact master-song latent-masking path that the current music-video workflow builds on.

---

## Update 5 — Persistent checkpoints, multi-clip workflows, and RAM-safe long-form assembly — 2026-08-15

### Main changes

- Added persistent per-clip H3 AV checkpoints with generation signatures, so unchanged clips can be reused from disk instead of being sampled again on every queue.
- Added crash/restart resume at clip boundaries for both music-video and AV-extension chains.
- Reworked long-form continuation so later clips and preview branches load saved checkpoint paths instead of depending on large cached latent tensors from earlier samplers.
- Added RAM-safe sequential final assembly that decodes one saved clip at a time, preserves **linear visual overlap blending at every seam**, and streams the result to ffmpeg.
- Added per-clip video previews and direct in-node preview of the already-written final MP4 without decoding the complete movie back into one giant `IMAGE` batch.
- Added a multi-clip AV Extension workflow that could either extend an existing video or generate a new T2V/I2V starter clip and then continue extending it.
- Expanded the master-song music-video workflow to a 20-slot checkpoint/resume architecture and shipped it preconfigured with the bundled six-clip reproducible demo.
- Fixed the master-song 40 Hz audio-grid boundary case where an exact picture-duration encode can be one latent token short.
- Reworked native H3 compatibility detection for Issue #7 so compatibility is checked behaviorally instead of by brittle source-string inspection.
- Consolidated and renamed example workflows, removed obsolete validation/demo duplicates, and moved workflow guidance into one `example_workflows/README.md` plus notes embedded directly in the workflows.

### 1. Persistent checkpoint gates and selective regeneration

The Update-5 long-form workflows used **persistent checkpoint gates** at clip boundaries.

Each completed H3 video/audio latent is saved to a fixed safetensors checkpoint together with a SHA-256 generation signature derived from the submitted graph that produced that clip. Before requesting the sampler again, the gate compares the current generation signature with the saved checkpoint:

- unchanged seed/prompt/upstream inputs → reuse the saved checkpoint without sampling;
- changed Clip N inputs → regenerate Clip N;
- later clips that depend on Clip N regenerate naturally;
- earlier unchanged clips remain reusable.

This behavior does not depend on ComfyUI retaining large latent tensors in RAM and continues to work after cache eviction or process restart as long as the matching checkpoint remains on disk.

The disk-backed path nodes also separate the small cached checkpoint-path string from the large H3 AV latent. Continuation and preview branches reload the saved latent from that path instead of forcing earlier samplers back into execution.

### 2. Crash-safe resume and checkpointed continuation

Update 5 added checkpoint/resume helpers for long H3 chains, including:

- **H3 Persistent Checkpoint Gate**;
- **H3 Checkpoint Load Path**;
- **H3 Checkpoint Tail Frames**;
- **H3 Resume / Saved AV Latent**;
- **H3 Resume / Live Tail Frames**;
- **H3 Resume / Live AV Latent**;
- **H3 Checkpoint Final Trigger**.

The music-video workflow can resume from completed clip checkpoints, and the AV Extension workflow can resume from either an existing generated extension or a saved generated starter clip.

### 3. RAM-safe assembly and previews

Added streamed assemblers for the long-form workflows:

- **H3 Assemble Checkpoints**;
- **H3 Assemble Extension Checkpoints**;
- **H3 Assemble Starter + Extension Checkpoints**.

Instead of repeatedly concatenating decoded `IMAGE` tensors into an ever-growing movie, the assemblers:

1. load one saved H3 checkpoint at a time;
2. decode only that clip;
3. retain only the small seam tail required for the next join;
4. apply the same linear visual overlap blend at every seam;
5. stream completed video/audio to ffmpeg.

This removes the cumulative decoded-image pattern that caused extremely large host-RAM usage in long chains.

The Update-5 long-form workflows also included clip-local VHS previews. The complete assembled MP4 is previewed directly from the encoded file through ComfyUI's native video-preview UI, so a full-movie `IMAGE` tensor is not recreated just for viewing.

### 4. Update-5 music-video workflow

The Update-5 music-video example was:

**`NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json`**

It provides:

- up to 20 clip/sampler slots;
- persistent checkpoint reuse and clip-boundary resume;
- exact master-song audio latent masking;
- optional reference images through the normal H3 reference-conditioning path;
- clip-local previews;
- streamed final assembly with linear visual blending at every seam;
- the untouched master song as the authoritative final soundtrack.

The shipped default reproduces the bundled six-clip demo using the included reference images, `I'll Know You by the Scar.wav`, the original authored Clip 1–6 prompts, and fixed versions of the original numerical seeds. Clips 7–20 remain generic template slots and are bypassed by default.

The full Director Prompt and exact audible-word alignment instructions are embedded in the large note at the top of the workflow itself. The director instructions require alignment to the **actual audible words in each calculated song slice**, rather than estimating lyric timing from transcript order or line count.

### 5. Update-5 multi-clip AV Extension workflow

The Update-5 multi-clip extension example was:

**`NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json`**

A single visible **H3 Extension Start Mode** choice controls the start source:

- **Start from existing video** — load a source video/audio clip and extend it;
- **start with T2V/I2V** — generate a new H3 starter first, then extend that generated clip.

For a generated starter:

- starter first frame off = T2V;
- starter first frame on + Load Image = I2V;
- the starter uses the FL2VA generation path;
- the starter is checkpointed before Extension 1;
- Extension 1 can resume directly from that saved starter checkpoint.

The same start-mode control also synchronizes the source-video branch so the unused VHS source loader is not submitted/validated when starting from T2V/I2V.

For both start modes:

- optional global reference images can guide the generated clips;
- up to six extension slots are available;
- every generated clip is checkpointed persistently;
- later extensions continue from the previous saved H3 AV latent tail;
- unchanged earlier clips can be reused from disk when regenerating a later clip;
- clip-local previews are available;
- final assembly is streamed checkpoint-by-checkpoint with linear visual blending at every seam.

### 6. Master-song 40 Hz audio-grid boundary fix

Fixed the boundary case where the H3 target audio grid can require one more 40 Hz latent step than an audio-VAE encode of the exact picture-duration waveform initially produces—for example, a target of 207 latent steps versus an initial 206-step encode.

The target H3 audio grid is now authoritative. The node uses enough **real master-song waveform** to cover the required grid and then crops the encoded latent to the exact target length. It does not solve the shortfall with fake silence, repeated latent tokens, or guessed padding.

### 7. Motion Context compatibility fix for Issue #7

The native H3 guide/Ref2VA compatibility check no longer depends on finding an exact Python source-code pattern inside `MiniMaxH3.extra_conds()`.

Update 5 probes the behavior of the active conditioning implementation instead:

- Simple Motion Context without references does not require Ref2VA merge behavior;
- image references require the required video-reference merge behavior;
- audio references additionally require the audio-reference merge behavior.

This avoids false failures after compatible ComfyUI refactors and improves coexistence with other H3 nodepacks that wrap or patch core H3 conditioning behavior, while still rejecting wrappers that actually break the required merge semantics.

### 8. Workflow and documentation cleanup

The example folder was simplified around clear architecture labels:

- **`NEW - Latent Masking - ...`** — the then-current latent-masking workflows;
- **`OLD - Motion Context - ...`** — legacy guide-conditioning examples;
- **`OLD - Hybrid - ...`** — mixed architectures retained for existing projects/experimentation;
- **`UTILITY - ...`** — helper/example graphs.

`OLD` means legacy architecture, not broken or unsupported.

Obsolete validation workflows, duplicate music-video examples, and per-workflow sidecar READMEs were removed. Workflow guidance was consolidated into **`example_workflows/README.md`** and notes embedded directly inside the relevant workflow JSON files.

For low-level H3 timing, masking, checkpoint/signature behavior, compatibility probing, and streamed assembly details, see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

---

## Update 6 — Long-form workflow rebuild, exact AV timing, and direct VHS streaming — 2026-08-17

The current `NEW - Music Video.json` restores the repository's reproducible six-clip bundled demo defaults: the two CC0 reference images, `I'll Know You by the Scar.wav`, authored Clip 1–6 prompts, fixed seeds, Ref2VA model selection, 1 MP 16:9 resolution, 15-second raw target, 8-step turbo setup, and six active clips.

### Main changes

- Rebuilt both current long-form workflows as **checkpoint-free direct-latent chains**.
- Renamed the two primary examples to **`NEW - AV Extension.json`** and **`NEW - Music Video.json`**, with utility and legacy examples moved under short `UTILITY -` / `OLD -` names.
- Removed `h3_checkpoint_resume.py` and the obsolete Update-5 checkpoint/resume registrations.
- Added controller-owned real ComfyUI bypass mode for AV Extension and Music Video generation/preview groups; neither current workflow requires rgthree for group control.
- Added direct single-pass final-output nodes that call VideoHelperSuite internally and do **not** materialize one complete final ComfyUI `IMAGE` movie tensor.
- Removed the superseded development-only live assemblers that returned a materialized full final `IMAGE` movie, so the known high-RAM path is not exposed as an alternative Update-6 node.
- Added a targeted VHS same-result inline-preview refresh for controller-managed preview groups.
- Added exact decoded-audio timebase conformance for small H3 audio-grid duration undershoots, replacing the previous silence-tail fallback at normal extension joins.
- Extended CPU/static regressions for live chains, controller ownership, workflow link integrity, direct-stream final wiring, exact AV boundaries, absolute PCM seam accounting, decoded-audio conformance, and exact master-song slice endpoints.

### AV Extension

- `Existing Video`, `T2V`, and `I2V / Custom Keyframes` share one controller and one live extension chain.
- T2V/I2V use `MiniMaxH3ReferenceToVideo`; I2V adds `H3 Custom Keyframes` at UI frame 1 / internal frame 0.
- Two image-reference slots feed the starter and all six extensions.
- Two optional audio-reference loaders feed the starter and all six extensions through one reroute per reference; both are bypassed by default.
- Generated clips pass their H3 AV latents directly into the next masked-context node.
- Audio masking keeps a hard preserved region plus a configurable half-cosine feather; the example defaults to 8 audio-latent ticks (0.2 s). Video masking remains hard.
- Protected AV context snaps to shared H3 video/audio boundaries `39 / 90 / 141 / 192 / ...`, rather than accepting video-only H3 runs that end between 40 Hz audio ticks.
- Final decoded-audio stitching removes duplicated context using **absolute timeline sample boundaries**. This avoids one-sample seam drift at rates such as 44.1 kHz.
- When H3 audio decode is only a few hundred samples short of the exact frame-derived clip duration, the whole decoded clip is time-conformed by the tiny required ratio **before** the duplicated context is cut. Normal grid undershoot no longer inserts a zero/silence tail at every extension.
- The public workflow now ends at `H3 Stream AV Extensions to VHS`. It decodes one generated clip at a time, keeps only the effective visual seam tail, streams completed frames directly into VHS, and assembles exact AV audio without allocating the old full final RGB movie tensor.

### Music Video

- The 20-slot Music Video workflow starts from a generated `MiniMaxH3ReferenceToVideo` clip and continues by passing each sampler's **video latent tail directly** to the next clip.
- The original master song remains authoritative: each clip receives the exact absolute master-song slice in its H3 audio latent with audio mask `0`, while final output receives the original master waveform unchanged.
- Song-slice endpoints are derived from **absolute timeline sample boundaries**, avoiding one-sample errors from independently rounded slice durations.
- The Director Prompt mirrors the workflow's timing math: requested seconds are quantized to 24-fps frames, snapped upward to the H3 `17k+5` frame grid, and each clip window is calculated from integer frame positions before converting to seconds.
- The final output does **not** trim generated picture to the master-song duration by default.
- Four global image-reference slots feed all 20 Ref2VA nodes; References 3 and 4 are bypassed by default.
- One Music Video controller owns Active Clips 1–20 and preview policy, plus one global KSampler selection and one global scheduler.
- `H3 Stream Final Music Video to VHS` decodes one live clip at a time, keeps only the local visual seam tail, streams frames directly into VHS, and passes the untouched master song to the final encode. The workflow connects its `VHS_FILENAMES` result to `H3 Final Stream Output Sink`; this keeps the streamer executable without giving its all-clips dependency chain the same output priority as the per-clip preview branches.

### Inline preview refresh

The current workflows keep ordinary VideoHelperSuite clip-preview nodes. VHS can overwrite a temp preview using the same filename/type/format as the previous result; its frontend may then treat the result as unchanged and leave the inline player stale/hidden even though `Open preview` reads the new complete file.

For H3-controller-managed preview groups, the frontend detects that exact same-result condition. After VHS's normal execution handler runs, it invalidates only the stored filename and invokes VHS's own forced `updateParameters(..., true)` path once. It does not directly force repeated video-source reloads.

### RAM/caching note

Direct final streaming removes the large full-movie RGB allocation from the public workflows. ComfyUI's own intermediate-result cache can still retain large latent outputs. Cache policy remains a ComfyUI startup/runtime choice and is not embedded in the workflow JSON.

### PR #15375 forward compatibility

ComfyUI PR #15375 changed its native H3 mask integration on 2026-08-15: mask-grid alignment moved from the earlier `process_denoise_mask` hook into `scale_latent_inpaint`, and the preprocessing hook was removed. Update 6 now recognizes both the earlier and current PR layouts. If the current PR architecture lands in ComfyUI, the local fallback self-retires instead of reinstalling the obsolete preprocessing hook. Fractional audio-mask values used by the 8-tick feather remain supported by the current PR design.

### Compatibility note

Update-5 disk-checkpoint long-form nodes are no longer registered in Update 6. Saved workflows containing those historical checkpoint nodes must be migrated to the current direct-latent workflows.

---

## Update 7 — Arbitrary-position latent inserts and keyframes — 2026-08-18

**Contributed by Reithan in [PR #3](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/3).**

Merged into `main` on **2026-08-30** after maintainer servicing and validation.

Update 7 adds **arbitrary-position latent inserts**, **hard-preserved masked keyframes**, and H3-aware AV noise-mask utilities.

Main changes:

- `H3 Existing Video Masked Context` gains an optional `insert_frame` input. The preserved segment can now be placed at any multiple-of-17 pixel frame in the target latent, not only the prefix. At 0 the behavior is byte-identical to the original prefix path. Multiples of 51 also align the audio clock exactly; non-51 inserts produce a sub-25 ms audio rounding that is logged as a warning.
- New `H3 Assemble Interior Insert` node — the required output path for interior inserts. The causal VAE cannot exactly round-trip an interior preserved region, so without this node interior inserts have no pixel-correct output. Splices the canonical source frames and audio back over the decoded H3 output at the preserved interval using exact AV accounting, matching the no-crossfade philosophy of `H3 Assemble Existing Video Extension`.
- New `H3 Custom Keyframes (Masked)` node — hard-preserves keyframes by writing each encoded still directly into the target AV latent and masking those steps from denoising, unlike the original soft node which uses conditioning rows. Phase-0 positions (1, 18, 35, ... in the default 1-based indexing) pin exactly one frame; interior positions pin the full containing latent step (up to 4 frames of static hold). Existing nested H3 AV masks are merged without changing their audio stream; a keyframe that overlaps an already protected video step raises instead of silently overwriting it. The JS keyframe-position widget now attaches to both node types.
- New `H3 Set AV Noise Mask` / `H3 Clear AV Noise Mask` utilities preserve H3's separate video/audio mask streams. Use these instead of stock `Set Latent Noise Mask` for H3 AV latents when audio-mask preservation matters.
- A non-multiple-of-17 `insert_frame` is snapped **down** to the nearest multiple of 17 with a warning; all other new failure modes raise with exact numeric accounting.

Merge-preparation servicing retained Reithan's feature attribution while adding safe existing-mask composition for hard keyframes, explicit overlap rejection, documentation/tooltips cleanup, and regression-test repairs.

---

## Update 8 — released 2026-08-30 — Fractional V2V, streaming, exact-grid and de-rope improvements

Step 1 adds `H3V2VGranularFractionalDenoise` (`H3 V2V Granular Fractional Denoise`), moving fractional V2V strength onto H3's per-stream denoise-mask path. "Granular" denotes the continuous `0..1` preserve-to-generate mask levels used instead of binary masking; "Fractional Denoise" describes their V2V effect. Near-1 mask precision compatibility is lazy and capability-probed: importing the repository does not patch ComfyUI, and executing the node patches only precision behaviors the live H3 core still lacks.

### Source-audio alignment correction

The first Step-1 build carried a defensive fallback that appended one zero **latent** audio step when the source encode was exactly one H3 audio tick shorter than the target. Deeper tracing showed that the latent shortfall was downstream of a more fundamental preprocessing mismatch.

Current ComfyUI's generic `VAE.encode()` center-crops every non-aligned input dimension down to its VAE downscale multiple before calling the model-specific encoder. For the H3 audio VAE that multiple is 800 PCM samples (25 ms at 32 kHz). The H3 audio encoder itself is designed to right-pad PCM upward to that 800-sample boundary, while the H3 AV target allocates its audio stream by rounding video duration to the nearest 40 Hz audio tick. Feeding an exact picture-duration waveform through the generic wrapper can therefore both shorten the encode and move its start because the generic crop is centered.

The V2V node now treats the **existing target H3 audio latent length as authoritative** and constructs an exact `target_audio_steps * samples_per_latent` PCM slice before encoding. This makes generic VAE center-cropping a no-op, keeps the selected source-audio origin aligned with the selected source-video origin, and naturally produces the required number of audio latents without post-encode latent padding or trimming. If the rounded 40 Hz grid extends fractionally past the final picture boundary, the node first uses real source-audio lookahead when available; only at an actual source endpoint can it add the small model-grid boundary pad in PCM space. A larger source-audio deficit remains an error.

This also fixes cases where the old path happened to return the correct number of audio tokens but still center-shifted the encoded waveform by several milliseconds. Any residual mismatch after exact-grid preparation is now reported as an audio-VAE wrapper/encoder contract error rather than silently changing latent length.

### Step 1c: repository-wide H3 audio-grid audit

The same generic-VAE center-crop risk was audited across every repo-side H3 audio encode. Exact-grid preparation is now shared in `h3_audio_grid.py` and used by all affected paths:

- **H3 V2V Granular Fractional Denoise**: start-aligned to the selected source-video interval; the target AV latent owns the audio-step count.
- **H3 Song Audio + Masked Video Context**: start-aligned to `clip_start_seconds`; the public `clip_audio` output remains exact picture-duration audio while the latent encode uses the exact target 40 Hz grid. The older retry/over-encode/latent-crop workaround is removed.
- **H3 Motion Context** when `context_audio` is VAE-encoded: the exact grid window is end/seam-aligned so continuation timing at the generated boundary does not move. Latent-to-latent audio context remains untouched.
- **H3 Masked AV Bridge**: start-source tails are end/seam-aligned and end-source heads are start/seam-aligned. Non-shared H3 video runs such as 56/73 frames no longer let generic center-cropping shift either bridge seam.

`H3 Existing Video Masked Context` was checked and is already mathematically safe because it snaps protected context to shared 24 fps / 40 Hz AV boundaries (`39 / 90 / 141 / 192 / ...`), where the PCM span is inherently an exact H3 audio-grid multiple. It now routes that exact span through the same strict shared encoder as a consistency/invariant check, without changing its timing behavior. `H3 Generated AV Masked Context`, latent-based Motion Context audio, decode/streaming nodes, and compatibility payload code do not audio-VAE encode source PCM and are therefore not affected by this issue.

After exact-grid PCM preparation, post-encode audio latent padding/trimming is considered an error-hiding anti-pattern. The shared encoder helper requires the audio VAE to return exactly the requested number of latent steps; otherwise it reports a wrapper/encoder contract mismatch.

### Step 1d: workaround-retirement audit

After the exact-grid source correction, the repository was re-audited for older
workarounds that had been compensating for PCM -> H3 audio length mismatches.
The following compensations are intentionally **retired**:

- V2V's one-audio-latent zero-pad fallback;
- Song Audio's extra-lookahead/retry encode followed by latent cropping;
- Masked AV Bridge post-encode head/tail latent trimming;
- Existing Video Masked Context post-encode latent trimming;
- stale latent-context "overhang compensation" bookkeeping that was no longer
  consumed by any caller.

All repo-owned PCM -> H3 audio encoding now goes through `h3_audio_grid.py` and
must return exactly the requested target-cell count. A mismatch after exact-grid
PCM preparation is an encoder/wrapper contract error, not something to repair by
changing latent length.

Several audio adjustments remain on purpose because they address **different
contracts** and are not workarounds for the generic VAE center crop:

- a partial PCM boundary cell may be zero-filled only at a genuine media edge
  when the authoritative rounded 40-Hz target grid extends slightly beyond the
  available picture-duration waveform;
- Motion Context may reduce a requested audio-context window when the supplied
  `context_audio` is genuinely shorter than the requested video context;
- decoded generated audio may be time-conformed by a tiny ratio to the exact
  24-fps picture duration before extension seams are cut;
- final assembly helpers still trim/pad decoded/public waveforms to explicit mux
  timeline lengths where required.

Those retained paths operate before encoding at a real source boundary or after
decoding/output assembly. They do not hide an H3 audio-VAE latent-count mismatch.

### Step 1e: bundled-workflow workaround audit

Every bundled workflow JSON was re-audited after the source-audio fix. No workflow
contains a remaining `546 -> 547` latent-pad, post-encode latent crop, retry encode,
or hard-coded 25-ms compensation. The audio operations that remain in the graphs
serve different purposes:

- the three legacy Motion Context workflows remove the duplicated protected audio
  overlap before `AudioConcat`; their decoded audio tails are now time-conformed in
  both directions to the exact frame-derived duration instead of assuming H3 audio
  always rounds long;
- `UTILITY - AV Bridge` keeps the decoded-bridge audio trim because the bridge output
  intentionally contains protected source-audio head and tail spans; Step 1m wires
  the trim start/duration to shared frame/FPS controls and computes them on H3's
  40-Hz audio grid;
- VHS `trim_to_audio` on temporary previews is only a mux/preview safeguard and does
  not modify PCM before H3 AudioVAE encoding;
- the music-video previews use each context node's exact picture-duration `clip_audio`,
  while the final output uses the untouched master-song waveform;
- generated-context paths copy H3 audio latents directly and never invoke the PCM
  audio encoder.

The audit also exposed a separate stale assumption in `MiniMaxH3MotionContextTrim`:
H3's 40-Hz audio target is rounded to an integer cell count, so a valid 24-fps clip
can decode a few milliseconds **shorter or longer** than the picture. The bundled
362-frame legacy clips are a round-down case (603 audio cells = 15.075 s versus
15.0833 s of picture), previously leaving an 8.33-ms undershoot per clip. The node
now performs the same tiny bounded timebase conformance in either direction, so that
drift cannot accumulate across `AudioConcat` chains.

One path intentionally remains outside repo control: optional `ref_audio_*` inputs on
stock `MiniMaxH3ReferenceToVideo` are encoded by ComfyUI core, not by this repository.
Those optional full-audio references therefore still inherit ComfyUI's generic H3
AudioVAE input-crop behavior until core fixes it. They are disabled/bypassed by
default in the bundled legacy workflows and are not used for the master-song audio
path.

### Step 1f: workflow title normalization

Current `NEW -` and `UTILITY -` workflows were normalized so ordinary nodes use their default ComfyUI titles. Stage prefixes remain only where they materially identify repeated generation stages. Graph topology, node IDs, links, and workflow settings were not changed as part of the title cleanup.

### Step 1g: V2V latent motion transfer example workflow

Added `NEW - V2V Latent Motion Transfer (with upscale and de-rope).json` to the bundled examples. The imported graph keeps the validated two-pass V2V/de-rope/upscale topology and settings unchanged: the original source anchors Pass 1, de-rope is generated only from the Pass-1 denoised/x0 result, the learned latent upscaler feeds the refinement stage, and exact recovery restores the original timeline.

For readability, custom sentence-style node titles were removed. Only the stage-defining Pass 1 / Pass 2 model loaders, Ref2VA nodes, samplers, and the Pass-1 fractional-V2V node retain short prefixes in front of their default node titles.

---


### Step 1q: dynamic de-rope seam continuation

Integrated the audited two-pass de-rope continuation path into Update 8. De-rope can fan the protected real-time seam into a variable-length smeared prefix, so a second-pass Motion Context guide cannot safely assume target frame 0.

- `H3 Motion Context` now has an appended `target_start` control. `0` preserves existing head-context behavior and its trim count; a positive value places the native guide on the target pixel-frame timeline and returns `trim_frames=0`. Timeline audio, when used, moves with the same offset and continues to use Update 8's exact H3 40 Hz input-grid encoding.
- Added `H3 Fan Recovered Context` (`MiniMaxH3FanRecoveredContext`). It consumes the exact `hold_map_used` from the same MAINodes `H3 Time Smear`, fans the recovered previous-clip tail onto that smear clock, repairs the low-resolution second-pass baseline, returns only the seam-nearest native-resolution guide frames, and calculates their dynamic interior start.
- Added `NEW - 2MP De-Rope Continuation - Working Example.json`, using core `ImageFromBatch`, existing `H3 Motion Context Trim`, and `H3 Assemble Existing Video Extension` for the rest of the seam path. No new pass-2 denoise mask or custom final assembler is introduced.
- This is distinct from Update 7's arbitrary `insert_frame` latent preservation: `target_start` places conditioning only, while `insert_frame` writes/protects source AV inside the latent and may require interior reassembly.

### Step 1r: bundled de-rope continuation reference asset

- Added `example_workflows/assets/derope_continuation_reference.png` and pointed `NEW - 2MP De-Rope Continuation - Working Example.json` at that exact bundled reference image.
- Added a regression covering both the workflow filename and the asset SHA-256 so the example cannot silently drift to another reference.
- Clarified `example_workflows/assets/LICENSE.md` so the older Update 4 CC0 dedication is not implicitly extended to this separately supplied Update 8 reference image.


### Step 1s: mark de-rope continuation as a working example

- Renamed the bundled workflow to `NEW - 2MP De-Rope Continuation - Working Example.json`.
- The graph itself is unchanged; the name and workflow guide now make explicit that this is a functional reference with exposed seam/timing plumbing rather than a tidied showcase graph.

### Step 1t: dated update references

- Dated the prominent Update 7 and Update 8 references in the main README, workflow guide, and technical architecture so users can tell when each numbered update landed.
- Update 8 is recorded as released on **2026-08-30**.
- Kept this file as the authoritative complete dated history.

### Step 1u: README recent-changes summary and technical chronology

- Removed the full Update 1–8 timeline from the main README; the README now focuses on important changes from the most recent updates and links here for the complete history.
- Expanded the Update 8 README section with the cache-isolation fix that prevents Active Clips / Active Extensions from invalidating existing sampler hashes, modular final AV/Music inputs, deterministic preview ordering, granular fractional V2V behavior, lazy H3 mask-precision compatibility, exact 40 Hz audio-grid handling, calculated AV-Bridge timing, and the dynamic de-rope continuation workflow/node.
- Added the complete dated Update 1–8 chronology to `TECHNICAL_ARCHITECTURE.md` as a technical orientation index while retaining this file as the authoritative detailed change log.
- Added regressions that require the README to stay timeline-free while preserving the important Update 8 bug-fix/feature summary.


### Step 1x: workflow-guide cleanup

- Removed obsolete references to superseded V2V packaging; Update 8 documents only the integrated `H3 V2V Granular Fractional Denoise` path and current dependencies.
- Expanded V2V tuning guidance: when source motion is not transferring strongly enough, reduce `global_strength` in small steps so slightly more source-latent structure remains as the motion/performance guide.
- Kept pass-2 denoise experimentation inside the workflow note itself instead of repeating that note's suggestion in the workflow README.
- Updated AV Extension guidance to reflect controller-owned activation: **Active Extensions** enables/bypasses managed extension groups automatically, so users do not manually enable extension groups in order.
- Simplified AV Bridge timing wording while retaining the shared-control and 40-Hz-grid behavior.


## Update 8 AV Extension source-preview follow-up — 2026-08-30

The Update 8 AV Extension workflow restores the normal `VHS_LoadVideo` source picker so Existing Video starts retain VideoHelperSuite's inline source preview, while keeping the silent-container workaround inside the MultiRef source-audio policy.

- `VHS_LoadVideo.audio` is intentionally left completely unconnected.
- `VHS_LoadVideo.video_info` identifies the selected source loader to `H3 Source Audio Policy`; the policy reads the selected filename from ComfyUI's queued `PROMPT`/`UNIQUE_ID` context and invokes ffmpeg directly.
- A genuine missing source-audio stream becomes exact-duration stereo silence. Other ffmpeg failures remain errors.
- **Regenerate with H3** keeps the full-source protected-video / generated-audio branch introduced in the local AV Extension work.
- The AV controller keeps its cache-isolated execution parameters so preview-only controller changes do not unnecessarily invalidate generation branches.
- Update 8's exact H3 40 Hz audio-grid encoding paths remain authoritative; the older local audio-encoding implementations were not copied back over them.

### Update 8 AV Extension final-stream modular inputs — Step 1j

`H3 Stream AV Extensions to VHS` no longer requires a contiguous fixed set
of extension sampler inputs. The node now has an **Input Count** widget (1–64)
and a browser-side **Update inputs** button that exposes exactly that many
`extension_N` latent sockets. Disconnected sockets are ignored; connected
extensions are streamed in socket-number order.

The bundled `NEW - AV Extension` workflow keeps its controller's
`active_extensions` parameter connected as an optional execution cap. This
preserves the controller behavior where inactive/bypassed higher extension
groups remain lazy, while custom workflows can leave the cap disconnected and
use the final streamer as a standalone modular multi-extension output node.

### Update 8 AV Extension preview ordering + bypass-safe final output — Step 1k

The bundled `NEW - AV Extension` graph now places a lazy preview barrier in front of the modular final streamer. The barrier requests only the highest enabled extension `VHS_VideoCombine` output inside the controller cap, and the final streamer resolves that gate before requesting its expensive source/extension latent inputs. This makes the last enabled extension preview complete before final all-clips assembly begins.

`H3 Stream AV Extensions to VHS` is now an intermediate-output node backed by the existing terminal sink pattern. The sink's filename input is optional, so bypassing or disconnecting the final streamer is a valid no-op instead of a prompt-validation failure. The streamer's optional same-type preview gate also gives ComfyUI a compatible passthrough when bypass mode rewires the node.

### Update 8 Music Video modular final output + cache-isolated Active Clips — Step 1l

The Music Video final-output path now shares the reusable modular-streaming behavior introduced for AV Extension while preserving the master-song timeline contract.

- `H3 Stream Final Music Video to VHS` now exposes an **Input Count** widget (1–64) plus the browser-side **Update inputs** action for dynamic `clip_N` latent sockets.
- Standalone workflows may leave trailing clip sockets disconnected. Connected Music Video clips must remain a contiguous prefix from Clip 1; middle holes are rejected because compacting later clips would move their visuals against the untouched master-song timeline.
- The bundled workflow keeps all 20 clip sampler links available and uses a cache-isolated internal `PrimitiveInt` as the active-clip cap.
- `H3 Music Video Controller.active_clips` is no longer a backend data dependency of the final streamer or any sampler. The frontend mirrors only the Active Clips widget into that internal parameter; changing Preview mode therefore does not invalidate final assembly, and changing Active Clips cannot change any existing sampler input hash.
- The shared `H3 Last Active VHS Preview Barrier` now supports both AV Extension and Music Video and waits on the highest enabled clip preview before final assembly requests clip latents.
- The preview barrier now also exposes **Input Count** plus **Update inputs**. The backend keeps its 64-slot compatibility range, but new/bundled workflows show only the configured preview sockets (6 for AV Extension, 20 for Music Video). Legacy saved barriers infer their previous visible preview span on load.
- The existing optional final VHS sink remains bypass-safe.


### Update 8 AV Bridge timing controls — Step 1m

`UTILITY - AV Bridge` now derives its post-sampler audio trim and visual overlap from shared timing controls instead of independent duplicated constants.

- A shared target-frame control drives the H3 Image-to-Video target length and the audio-grid duration calculation.
- A shared preserve-frame control drives `H3 Masked AV Bridge`; the bridge's validated `preserve_frames` output drives both visual overlap nodes.
- A shared FPS control drives both bridge source-FPS inputs, final `CreateVideo`, and the trim calculations.
- `ComfyMathExpression` nodes calculate the decoded generated-audio window on the H3 40-Hz grid: `round(preserve / fps * 40) / 40` for the trim start and `(round(target / fps * 40) - 2 * round(preserve / fps * 40)) / 40` for the generated middle duration.
- The default `192 / 39 / 24` values therefore still evaluate to exactly `1.625 s` and `4.75 s`, but changing the timing controls no longer requires manually updating multiple unrelated widgets.

### Update 8 rebase onto merged Update 7 (PR #3) — Step 1n

Update 8 is now based on the repository's merged Update 7 from Reithan's PR #3 instead of carrying a parallel numbering line.

- Preserved Update 7's arbitrary `insert_frame` support, `H3 Assemble Interior Insert`, hard-masked keyframes, and H3 Set/Clear AV Noise Mask utilities.
- Preserved the maintainer servicing added before the Update 7 merge: nested H3 AV mask composition for hard keyframes, explicit protected-step overlap rejection, corrected snap-down wording, and graph-role-based regressions.
- Reconciled `H3 Existing Video Masked Context` with Update 8's source-audio policy and strict exact-grid PCM encoder. A 39-frame preserved segment still enters the audio VAE as exactly 65 H3 ticks / 52,000 PCM samples at 32 kHz even when placed at an interior frame such as 17. Non-51-frame insert offsets quantize only the destination audio-step start and keep the existing warning; multiples of 51 align both clocks exactly.
- Kept `H3 Assemble Interior Insert` on exact frame-derived PCM splice boundaries after decode, so H3 latent-grid placement quantization does not accumulate into final-output audio drift.
- Kept all Update 8 AV Extension, Music Video, V2V/de-rope/upscale, cache-isolation, preview-barrier, bypass-safe final-output, and calculated AV-Bridge timing changes.
- Renamed the former Update-7 WIP test/docs references to Update 8 while leaving Reithan's merged feature history as Update 7.

The Step-1n combined CPU/static regression suite passes 167 checks before final package/checksum regeneration.

### Step 1v: README V2V workflow highlight

- Added a dedicated Update 8 README highlight for `NEW - V2V Latent Motion Transfer (with upscale and de-rope)` before the granular-denoise implementation bullet.
- The README now explains that the workflow transfers motion/timing/performance from a source video, applies generated-x0 de-rope, learned latent upscaling, a second refinement pass, and exact timeline recovery.
- Clarified that the workflow uses `H3 V2V Granular Fractional Denoise` specifically to retain a faint source-latent motion guide while leaving enough denoise freedom to replace identity and appearance.
- Runtime code and workflow graphs are unchanged.


### Step 1y: AV Extension Prompt Director note

- Embedded the supplied **MINIMAX H3 AV EXTENSION — PROMPT DIRECTOR** directly at the top of `NEW - AV Extension.json` as a workflow Note for copy/paste prompt planning guidance.
- Removed the stale workflow-note instruction to manually enable extensions in order; the bundled controller owns managed extension-group activation through **Active Extensions**.
