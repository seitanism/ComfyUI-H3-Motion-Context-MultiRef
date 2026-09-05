# Technical Architecture

This document explains the implementation behind the workflows in this repository. It is intentionally more technical than the main README.

If you only want to choose and run a workflow, start with [README.md](README.md) and [example_workflows/README.md](example_workflows/README.md).

> **Update 9 (2026-09-05):** current examples use constant 24 fps input/output. Guide conditioning and latent preservation remain separate families. AV Bridge contexts are restricted to 39 + 51*k frames; targets retain 5 + 17*k with exact final PCM assembly. Sections 14–19 are historical Update-5 checkpoint architecture only; those checkpoint/resume nodes are not registered. See [workflow prerequisites](WORKFLOW_PREREQUISITES.md) and the detailed [keyframes/inserts guide](KEYFRAMES_AND_INSERTS.md).

### Dated update chronology

- Update 1 — **2026-08-10** — custom keyframes
- Update 2 — **2026-08-12** — existing-video extension and compatibility improvements
- Update 3 — **2026-08-14** — per-token video/audio latent masking; the corresponding H3 core support was contributed upstream in [ComfyUI PR #15375](https://github.com/Comfy-Org/ComfyUI/pull/15375), “Support per-token video and audio latent noise masks on MiniMax-H3”
- Update 4 — **2026-08-14** — exact song-latent masking
- Update 5 — **2026-08-15** — persistent checkpoints and long-form assembly
- Update 6 — **2026-08-17** — direct-latent long-form workflows, exact AV timing, and VHS streaming
- Update 7 — **2026-08-18** changelog/PR date; merged to `main` **2026-08-30** — arbitrary inserts, hard-masked keyframes, and H3 AV mask utilities ([PR #3](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/3) by Reithan)
- Update 8 — **2026-08-30** — granular fractional V2V, exact audio-grid paths, modular streaming/cache fixes, calculated bridge timing, and dynamic de-rope seam continuation
- Update 9 — **2026-09-05** — audit repairs, strict current-workflow timing, dated output folders, lazy audio-crop/velocity compatibility, and expanded node guidance

`MODIFICATIONS.md` is the authoritative full change log; this chronology is only a technical orientation index.

---

## 1. Architecture overview

The repository contains two different continuation families.

### Native guide conditioning

The previous clip is represented as **native H3 guide conditioning**. The target H3 latent remains a fresh generation target; previous motion/audio is supplied as conditioning that tells H3 what is already happening.

Main node:

- `H3 Motion Context`

Used by the new 2MP de-rope continuation example and retained `OLD - Motion Context - ...` workflows. Guide conditioning is current functionality; the OLD label describes those earlier example graphs.

### Current latent masking

Known video/audio content is written directly into the H3 target latent. A per-token noise mask marks the known region as protected and the unknown region as generative.

Main nodes:

- `H3 Existing Video Masked Context`
- `H3 Generated AV Masked Context`
- `H3 Masked AV Bridge`
- `H3 Song Audio + Masked Video Context`

Current workflow examples:

- `NEW - AV Extension.json`
- `NEW - Music Video.json`

The two families can coexist in the same repository because they solve continuation in different ways.

---

## 2. MiniMax H3 joint AV latent

MiniMax H3 works with a joint audiovisual latent. In this repository it is treated as two streams inside the ComfyUI `LATENT` object:

- video latent: approximately `[B, C, T, H, W]`;
- audio latent: approximately `[B, C, 2, T_audio]`.

Current ComfyUI H3 represents these together using a nested tensor-like structure.

The custom nodes therefore avoid assuming that `latent["samples"]` is one ordinary tensor. Helper functions explicitly unpack the video and audio streams.

This matters for:

- target-latent masking;
- historical checkpoint serialization;
- direct generated-latent continuation;
- master-song audio replacement.

---

## 3. Video and audio clocks

H3 video output is treated as **24 fps**.

H3 audio latents run at **40 latent steps per second**.

Therefore:

```text
40 / 24 = 5 / 3
```

One video frame is not one audio latent step.

### Video-VAE temporal pattern

The H3 video latent uses the repeating pixel-frame coverage pattern:

```text
1, 4, 4, 4, 4
```

The repository exposes this internally as:

```python
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
```

A complete native temporal video run therefore follows the pixel-frame sequence:

```text
5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, ...
```

which is:

```text
17k + 5
```

for integer `k >= 0`.

### Exact AV boundaries

Some native video runs also land exactly on the 40 Hz audio grid.

Examples:

```text
39 frames  = 1.625 s = 65 audio steps
90 frames  = 3.750 s = 150 audio steps
141 frames = 5.875 s = 235 audio steps
192 frames = 8.000 s = 320 audio steps
```

This is why **39 frames** is the default continuation context throughout the current workflows: it is a valid H3 video-VAE run and an exact video/audio timing boundary.

---

For Update-6 masked AV continuation, a protected context must be both a valid H3 video-VAE run and an exact 40 Hz audio-latent boundary. The shared sequence is `39, 90, 141, 192, 243, ...` frames. Requested context values are snapped downward to the largest shared boundary that fits the available source and target. This avoids fractional audio endpoints such as 73 video frames (`121.666...` audio ticks).

Final decoded-audio stitching uses **absolute frame-to-sample boundaries** rather than repeatedly rounding a relative context duration. At 44.1 kHz, for example, two 39-frame seams can legitimately require 71662 and 71663 samples at different absolute timeline positions.

## 4. Why some target clips have a rounded audio tail

A full H3 target does not always end on an exact 40 Hz audio boundary.

Example:

```text
124 video frames / 24 fps = 5.166666... s
5.166666... s * 40 Hz = 206.666666... audio steps
```

The target AV latent may therefore contain:

```text
207 audio latent steps
```

The target latent length is authoritative.

### Pre-encode PCM alignment and the generic VAE crop

There is a second, easy-to-miss boundary at the **PCM input to the audio VAE**. Current ComfyUI's generic `VAE.encode()` wrapper center-crops non-aligned input dimensions down to the VAE downscale multiple before the model-specific encoder runs. For the current H3 audio VAE, one latent cell is 800 PCM samples at 32 kHz, i.e. 25 ms / one 40 Hz step. The H3 audio encoder itself can right-pad PCM to that boundary, but the generic wrapper's center-crop happens first. Feeding it an exact picture-duration waveform that is not already on the H3 audio grid can therefore both shorten the encode and shift the waveform origin by a few milliseconds. This upstream behavior was reported in [ComfyUI issue #15970](https://github.com/Comfy-Org/ComfyUI/issues/15970). A narrow core fix was then submitted in [ComfyUI PR #15972](https://github.com/Comfy-Org/ComfyUI/pull/15972): disable the generic input crop for `MiniMaxH3AudioVAE` so the model-specific encoder receives the complete waveform and can perform its own right-padding. That preserves the waveform origin and restores the expected H3 audio-latent length without changing generic VAE behavior for other models.

The repository avoids that wrapper crop rather than compensating afterward: repo-owned PCM -> H3 audio encode paths first construct an **exact `target_audio_steps * samples_per_latent` PCM window with the correct semantic anchor**, then call the VAE. Post-encode latent padding/cropping is not used to repair length mismatches.

This distinction is important for the master-song workflow because an audio VAE given only the exact picture-duration waveform can, on some encode paths, return 206 steps while the H3 target contains 207.

`H3 Song Audio + Masked Video Context` handles this by:

1. keeping its public `clip_audio` output at the exact picture duration;
2. taking the target H3 audio latent length as authoritative;
3. converting that target length to an exact PCM grid (`target_steps * samples_per_latent`);
4. taking that grid-sized master-song interval starting exactly at `clip_start_seconds`, using real song lookahead when the rounded 40 Hz endpoint extends beyond the final picture frame and only allowing the small expected PCM boundary pad at a real source endpoint;
5. giving the generic VAE wrapper an already aligned PCM length, so its center-crop path is a no-op;
6. requiring the encoder to return exactly the target number of audio latents.

The implementation does not fabricate, repeat, pad, or trim latent audio tokens. A mismatch after exact-grid PCM preparation is treated as a wrapper/encoder contract error.

The Update-8 exact-grid/workaround-retirement audit also removed the earlier retry/over-encode-and-crop paths and stale latent-context overhang bookkeeping. PCM endpoint padding and post-decode timeline conformance remain intentionally separate: they resolve physical media/timeline boundaries, not encoder output length.

### Decoded-audio timebase conformance

The inverse problem also appears after H3 audio-VAE decode. A video-valid clip such as 362 frames does not represent an integer number of 40 Hz audio ticks, so the decoded waveform can be slightly **shorter or longer** than the exact picture timeline depending on which side of the rounded 40 Hz boundary the frame count lands.

For small grid-sized mismatches, `existing_video_extension.py` uses `_conform_waveform_length()` to resample the decoded waveform by the tiny rational ratio required to reach the exact frame-derived sample span. The correction works in either direction and happens **before** the protected-context samples are removed. The seam cut itself then uses absolute frame-to-sample boundaries.

The conformance path is deliberately bounded (`max_fractional_change=0.005` by default). A larger mismatch is treated as a real error rather than being silently stretched. The implementation does not append a zero/silence tail for these normal H3 grid-sized mismatches.

---

## 5. Noise-mask semantics

The latent-masking workflows use per-stream H3 noise masks.

Conceptually:

```text
0 = preserve / do not denoise this known token
1 = denoise / generate this token
```

For a normal continuation:

```text
VIDEO: [ protected previous context ][ generate future ... ]
MASK:  [ 0 0 0 0 ...               ][ 1 1 1 1 ...      ]
```

For an AV continuation:

```text
VIDEO: [ protected AV prefix ][ future ]
AUDIO: [ protected AV prefix ][ future ]
```

both streams can have a protected prefix.

For the master-song music-video workflow:

```text
VIDEO MASK: protected previous visual prefix = 0, future = 1
AUDIO MASK: entire master-song audio latent = 0
```

The song is therefore treated as authoritative target content rather than something H3 must reconstruct through denoising.

### Hard-preserved keyframes: `H3 Custom Keyframes (Masked)`

Class:

```text
MiniMaxH3CustomKeyframesMasked
```

The masked keyframes node applies the same mask semantics to still-image anchors: it VAE-encodes each still, writes it directly into the target AV latent at its quantized latent step, and zeros the video noise mask at exactly those steps so the sampler never denoises them. Audio retains the incoming audio mask. If none exists, it defaults to all-generate.

Hard vs soft:

| | Soft (`H3 Custom Keyframes`) | Hard (`H3 Custom Keyframes (Masked)`) |
|---|---|---|
| Mechanism | conditioning rows | latent write + mask |
| Adherence | H3 uses image as suggestion | latent-step protection; decoded pixels may differ |
| Input | `conditioning` | `latent` + `vae` |
| Output | `CONDITIONING` | `LATENT` |
| Audio | follows conditioning | existing audio mask retained; otherwise all-generate |

Phase-0 vs interior positions: phase-0 positions (`1, 18, 35, 52, ...` in the default 1-based indexing mode; `0, 17, 34, 51, ...` in 0-based mode) map to the single still-token step (1 pinned frame). Interior positions map to the containing 4-frame latent step, protecting the whole containing step (up to 4 frames; a static decoded hold is not guaranteed); the node logs the frame, the full pinned span, and the nearest phase-0 positions when this happens.

Duplicate positions that resolve to the same latent step after quantization raise with both slot numbers named. If the incoming latent already carries a **nested H3 AV `noise_mask`**, the node preserves that mask and unions the new hard-keyframe protection into the video stream while leaving the audio mask unchanged. A hard keyframe may not claim a video latent step that is already protected (including fractional protection); that overlap raises with the keyframe position and latent step named. Plain/single-tensor masks are rejected with guidance to use the H3 AV mask utilities below.
When combining this with `H3 Existing Video Masked Context`, put `H3 Custom Keyframes (Masked)` **after** the existing-video context node so it can merge into that node's AV mask.

Both `H3 Custom Keyframes` and `H3 Custom Keyframes (Masked)` share the same JS keyframe-position widget.

### H3 AV noise-mask utilities

`H3 Set AV Noise Mask` and `H3 Clear AV Noise Mask` are the supported mask-editing utilities for MiniMax H3 AV latents. H3 carries separate video and audio mask streams inside a nested mask. **Do not use ComfyUI's stock `Set Latent Noise Mask` when you need to preserve H3 AV mask semantics**: a stock single-tensor mask can replace the nested structure, leaving the audio mask stream absent and causing audio that was meant to stay protected to regenerate.

`H3 Set AV Noise Mask` accepts a video mask, an audio mask, or both. An omitted stream keeps the latent's existing H3 mask stream when present; otherwise it defaults to all-generate (all ones). `H3 Clear AV Noise Mask` removes the complete nested mask without changing latent samples, which is useful before rebuilding one or both streams deliberately.

---

## 6. `H3 Existing Video Masked Context`

Class:

```text
MiniMaxH3ExistingVideoMaskedContext
```

Purpose: start a latent-masked H3 continuation from a normal decoded video/audio source for which no original H3 sampler latent exists.

The arbitrary-placement extension to this node was contributed by **Reithan in [PR #3](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/3), “Implement arbitrary inserts,”** and released as Update 7. Update 8 keeps those insert semantics while routing the source-audio encode through the repository's strict exact-grid PCM path.

The node:

1. normalizes source timing to the H3 24 fps timeline;
2. selects the final requested context window;
3. snaps a masked prefix to a valid H3 video run where required;
4. resizes/crops source frames to the target H3 geometry;
5. VAE-encodes the source video tail;
6. takes the matching physical audio interval and audio-VAE encodes it;
7. writes both encoded streams into the fresh target AV latent at `insert_frame`;
8. creates video/audio masks protecting that segment;
9. leaves the rest of the target denoisable.

This node is normally used only for the **first** generated extension after an arbitrary uploaded video.

### `insert_frame` — the 17/51 grid rule

`insert_frame` specifies the pixel frame where the preserved segment begins in the target latent. At `0` the behavior is byte-identical to the original prefix path. Valid positions are multiples of 17 (the H3 latent phase grid: one still-token step + four 4-frame steps = 17 frames). A non-multiple is snapped down to the nearest multiple of 17, with the snap logged.

Multiples of 51 additionally land on a joint video+audio clock boundary (lcm of the 17-frame video grid and the 3-frame audio-frame boundary at 24 fps / 40 Hz). Using a non-multiple-of-51 insert introduces a sub-25 ms audio rounding that is logged as a warning.

```text
insert_frame 0:    prefix, trim_frames = n         (original behavior, byte-identical)
insert_frame 17:   interior, audio rounded by ~8 ms (log warning)
insert_frame 51:   interior, exact AV boundary
insert_frame 102:  interior, exact AV boundary
...
```

The node returns four outputs: `latent`, `trim_frames`, `insert_frame`, `preserved_frames`. Wire `insert_frame` and `preserved_frames` to `H3 Assemble Interior Insert`.

The preserved **duration** remains exact-grid even for an interior placement. With the default 39-frame context, the source span is always 65 H3 audio steps = 52,000 PCM samples at 32 kHz before encode. At `insert_frame=17`, only the destination audio-step start is quantized onto H3's 40 Hz grid; the source span itself is not shortened or center-cropped. The final assembler then restores source audio on exact frame-derived PCM boundaries, so this latent-placement quantization cannot accumulate into final-output drift.

### Interior inserts: `H3 Assemble Interior Insert`

Class:

```text
MiniMaxH3AssembleInterior
```

**For interior inserts (`insert_frame > 0`), this node is the required output path.** The causal VAE cannot exactly round-trip interior latent regions, so the decoded output needs pixel-correct source splicing. It splices canonical source frames and audio back over the interior preserved interval in the decoded H3 output.

Wire inputs:

- `continuation_images` / `continuation_audio` — the full decoded H3 output (all frames);
- `source_frames` / `source_audio` / `source_fps` — the same source used in the context node;
- `insert_frame` / `preserved_frames` — wire directly from the context node's new outputs;
- `fps` / `crop` — must match the context node.

The node uses the same CFR index map and resize call the context node used, so the spliced pixels are identical to what the mask was built around. Hard-cut splice with exact AV accounting; no crossfade, matching `H3 Assemble Existing Video Extension`.

---

## 7. `H3 Generated AV Masked Context`

Class:

```text
MiniMaxH3GeneratedAVMaskedContext
```

Purpose: continue from a previous **generated H3 clip** without decoding and re-encoding its continuation context.

The previous sampler already contains the H3 video/audio latent representation. The node therefore copies the previous clip's final valid AV latent run directly into the next target's prefix.

For a 39-frame continuation window, it derives the corresponding H3 video-latent and audio-latent lengths, copies the tail, and protects the copied prefix with mask `0`.

Advantages:

- no previous-clip video decode → VAE encode round trip for continuation conditioning;
- no previous-clip audio decode → audio VAE encode round trip;
- smaller continuation path;
- exact use of the generated H3 representation.

Source and target latent geometry must match, so chained clips should keep the same H3 resolution/model configuration.

---

## 8. `H3 Masked AV Bridge`

Class:

```text
MiniMaxH3MaskedAVBridge
```

Purpose: generate a missing middle section between two known audiovisual endpoints.

The node places:

- the end of source A into the beginning of the target latent;
- the beginning of source B into the end of the target latent;
- mask `0` on the known endpoint regions;
- mask `1` over the unknown middle.

H3 then generates only the middle region.

The delivered workflow still uses visual overlap treatment at the final source/generated joins because decoded source pixels and VAE-reconstructed generated endpoint pixels can differ slightly even when they represent the same content.

Update 9 accepts only shared AV context lengths: **39 + 51*k** (39, 90, 141, 192, …). The target remains a full H3 run **5 + 17*k**, and must leave a nonempty middle after subtracting both contexts. A 56-frame context is rejected even though it is a valid full video run.

`H3 AV Bridge Timing` validates controls before sampling. `H3 Assemble Bridge Audio` removes the protected head/tail at exact 40 Hz boundaries, then time-conforms only the generated middle to its absolute 24 fps sample span. This is necessary for targets such as 107 or 124, whose latent-audio endpoints round to whole cells. It fits the source intervals and concatenates on absolute sample boundaries, avoiding accumulated relative-rounding drift. Larger-than-grid timing discrepancies raise. This replaces the old shared-FPS/math/TrimAudioDuration/AudioConcat chain. Current source validation checks both VHS source and loaded FPS; output remains 24 fps.

---

## 9. Master-song latent masking

Class:

```text
MiniMaxH3SongMaskedAVContext
```

Display name:

```text
H3 Song Audio + Masked Video Context
```

Purpose: make one original song the authoritative audio timeline for a multi-clip H3 music video.

For each clip the node:

1. inspects the target H3 AV latent;
2. determines the target video duration and target audio-grid length;
3. selects the picture-duration master-song interval from `clip_start_seconds` for the public audio output;
4. resamples to the H3 audio-VAE input rate when required;
5. constructs a second, start-aligned PCM interval whose length is exactly the target H3 audio grid;
6. audio-VAE encodes that exact-grid interval and requires an exact target-length latent result;
7. writes that audio latent into the target audio stream;
8. sets the complete audio denoise mask to `0`;
9. optionally inserts/protects previous visual context at the beginning of the target video stream;
10. leaves the new visual region denoisable.

The final delivered music video uses the original master song, not a chain of decoded H3-generated audio clips.

### Clip start timing

For equal-sized raw H3 clips with a protected visual overlap:

```text
raw clip duration = raw_frame_count / 24
context duration  = context_frames / 24
new timeline advance = (raw_frame_count - context_frames) / 24
```

Therefore:

```text
Clip N start = (N - 1) * new timeline advance
```

The song slices overlap by the same physical time interval as the visual continuation context.

---

## 10. Reference images and Ref2VA

Reference images and latent continuation have different jobs.

A reference image primarily defines stable identity/appearance.

The protected continuation context defines the current physical state at the clip boundary:

- pose;
- current expression;
- camera framing;
- camera trajectory;
- motion;
- lighting state;
- environment/object state.

The current workflows therefore allow native `MiniMaxH3ReferenceToVideo` conditioning to coexist with latent-masked continuation.

`H3 Optional Reference Image` is a lazy helper node used by the multi-clip extension workflow. Disabled slots return no image; enabled slots request the connected image and pass it into the normal H3 reference input.

---

## 11. Native `H3 Motion Context`

Class:

```text
MiniMaxH3MotionContext
```

The legacy workflows do **not** insert previous content into the target latent. Instead they create native H3 guide/keyframe conditioning.

Current classic Motion Context uses ComfyUI's native H3 guide architecture.

Important concepts:

- previous visual frames become native H3 video guide data;
- timeline audio can become native H3 audio-guide data;
- ordinary Ref2VA references remain in their normal reference-conditioning path;
- ComfyUI combines native guide and reference payloads.

### Native video run vs per-frame fallback

For an exact H3 video run such as 39 frames, Motion Context can VAE-encode the whole temporal guide in one call.

For an arbitrary off-grid context length, the implementation can fall back to per-frame/still-guide representation so the requested endpoint is preserved rather than silently changing the context length.

This is different from target-latent masking, where the protected prefix itself must map cleanly into H3's target temporal latent.

When `context_audio` is supplied instead of a previous H3 AV latent, Motion Context must also respect the H3 audio grid before VAE encoding. The requested audio guide length is converted to the nearest 40 Hz latent-step count, then an exact PCM grid window is taken from the **tail** of the supplied audio. Tail/end alignment is intentional: for an off-shared-boundary context such as 56 frames, the unavoidable sub-token timing difference is placed at the older outer edge while the continuation seam remains fixed. Exact shared boundaries such as 39/90/141 frames are unchanged. When `context_latent` is supplied, audio is sliced directly in latent space and this PCM issue does not apply.

---


### Interior Motion Context placement for de-rope seams

Update 8 (2026-08-30) appends `target_start` to `H3 Motion Context`. The value is a pixel-frame position on the current target timeline. At `0`, behavior is unchanged: the guide occupies the head and its covered span is returned as `trim_frames`. At a positive offset, the native guide is placed inside the target and `trim_frames=0`, because the guide is conditioning rather than a duplicated delivery prefix.

This is intentionally different from Update 7 (2026-08-18), which introduced `insert_frame` on `H3 Existing Video Masked Context`. `target_start` changes native conditioning coordinates only; it does not write source pixels/audio into the target latent and does not require `H3 Assemble Interior Insert`. If timeline-mode Motion Context audio is supplied, its guide coordinate is shifted with the visual guide while the encoded audio itself still follows Update 8's exact 40 Hz grid contract.

### Hold-map-aware recovered seam fan

`H3 Fan Recovered Context` bridges a recovered real-time seam into a de-roped second pass. It reads the positive integer hold factors from the exact `hold_map_used` emitted by the same `H3 Time Smear`, fans the previous clip's recovered tail with those factors, and overwrites the corresponding beginning of the low-resolution smeared baseline.

For `N` seam frames and a fanned span `F`, the helper returns only the final `N` frames of that fanned seam as the native-resolution Motion Context guide and returns `F - N` as its target start. This keeps the guide nearest the true continuation boundary without hard-coding an offset and avoids materializing a second full-resolution fanned prefix.

The bundled `NEW - 2MP De-Rope Continuation - Working Example.json` uses this path without a pass-2 denoise mask:

```text
first continuation pass -> x0 decode -> Jerk Oracle / Time Smear
    -> fan recovered previous-clip seam onto the same smear clock
    -> encode repaired smear -> latent upscale -> H3 V2V Init
    -> native Motion Context at dynamic interior target_start
    -> second-pass refinement -> Exact Recover
    -> existing Motion Context Trim / exact AV assembly
```

## 12. Motion Context trimming and legacy assembly

A classic guide-based continuation repeats the guided visual prefix at the beginning of the generated output.

`H3 Motion Context Trim` removes that repeated region for delivery and can keep a smaller visual overlap specifically for a final blend.

When decoded audio is connected, the trim path also enforces the exact picture timebase. H3's rounded 40 Hz audio grid can decode a valid clip a few milliseconds short or long (for example, 362 frames at 24 fps corresponds to 603 H3 audio cells = 15.075 s, versus 15.0833... s of picture). Before removing the duplicated protected prefix, the node performs the same small bounded timebase conformance used by current AV assembly, then cuts audio on the exact frame-derived sample boundary. This prevents per-clip undershoot/overshoot from accumulating across legacy `AudioConcat` chains. It is a **post-decode timeline correction**, separate from the pre-encode exact-grid PCM preparation described above.

The legacy workflows historically used KJNodes `ImageBatchExtendWithOverlap` to accumulate and blend clips.

That visual blend is valid; the long-form memory problem came from the **cumulative tensor topology**:

```text
clip 1 + clip 2 -> large IMAGE batch
large batch + clip 3 -> larger IMAGE batch
larger batch + clip 4 -> ...
```

The historical Update-5 checkpoint assemblers preserved the blend operation while avoiding the ever-growing ComfyUI image batch.

---

## 13. Current Update-6 direct VHS final streaming — 2026-08-17

The current long-form workflows use direct single-pass streaming backed by VideoHelperSuite:

- `H3 Stream Final AV Extension to VHS` (`MiniMaxH3StreamLiveExtensionAVToVHS`) is the AV Extension output node.
- `H3 Stream Final Music Video to VHS` (`MiniMaxH3StreamLiveMusicVideoToVHS`) performs the Music Video encode and returns `VHS_FILENAMES`; `H3 Final Stream Output Sink` (`MiniMaxH3FinalizeVHSOutput`) is the terminal output node for that workflow.

The Music Video streamer is intentionally an intermediate-output node. Keeping the tiny filename sink one graph step after it prevents the streamer's all-clips dependency chain from competing with each clip-local `VAEDecode -> VHS Video Combine` preview as an equally direct output dependency.

A normal ComfyUI `IMAGE` output is a materialized tensor. For long H3 timelines, a complete RGB float32 movie can therefore consume tens of GiB before the encoder even starts. The Update-6 streaming nodes avoid exposing the final movie as an `IMAGE` output.

The internal frame source is a one-shot sequence tailored to the access pattern used by `VHS_VideoCombine`: VHS asks for `len(images)`, probes `images[0]` for dimensions, and then iterates the source. The sequence primes the generator once so Clip 1 is not decoded twice. Update 9 clones that first frame into independent storage: neither VHS’s first-image local nor the one-shot iterator can retain the entire first decoded batch through a frame view.

Final filename allocation is intentionally left to VHS. The H3 streaming layer passes through `filename_prefix` but does not construct, overwrite, rename, or delete final output paths. VHS therefore owns its normal numbered-output counter across repeated runs. This is a release invariant: final H3 output code must not return to a fixed filename that replaces the previous run.

For generated clips, the streamer:

1. decodes one H3 video latent to CPU float32;
2. computes the same effective overlap rule as the old assembler (`min(requested_overlap, effective_context, written_timeline)`);
3. keeps only the tail needed for the next seam;
4. linearly blends that retained tail against the matching decoded context of the next clip;
5. yields completed frames to VHS;
6. releases the decoded clip before continuing.

For an uploaded Existing Video source, source frames are CFR-indexed to 24 fps and resized in small chunks; only the source seam tail is retained by the streaming iterator before generated Extension 1. The original source IMAGE tensor may still remain alive in ComfyUI’s graph cache.

The AV streamer builds the final audio waveform separately using exact absolute sample boundaries and decoded-audio timebase conformance. Each extension owns its full protected audio overlap in the final buffer, preserving the generation-side feather. At the default 39 frames (65 audio ticks), 57 ticks stay hard-protected and only the final 8 ticks (0.2 s) gradually receive denoising. The whole 1.625-second context is not regenerated. Because assembly copies the extension decode, VAE reconstruction and bounded timebase conformance can still cause small differences outside the feather; this is not a promise of bit-identical original PCM. The Music Video streamer passes the original master waveform through unchanged.

### Inline VHS preview refresh

The current controllers own real ComfyUI bypass mode for preview groups. VideoHelperSuite can return the same temp filename/type/format after overwriting a completed preview. Its normal `updateParameters()` path may treat those parameters as unchanged and skip the visible inline refresh.

For controller-managed preview groups only, the H3 frontend detects that exact same-result condition. After VHS's normal execution handler has run, it invalidates only the stored filename and calls VHS's own `updateParameters(params, true)` once. It does **not** directly call the video widget's `updateSource()` method, avoiding the partial/truncated reload behavior seen with forced source refreshes.

### ComfyUI result caching

The streaming nodes bound the RAM used by final RGB assembly, but they do not control ComfyUI's node-result cache. ComfyUI's default RAM-pressure cache can retain sampler/latent results between nodes and runs until its own pressure thresholds decide to evict them. This can make total process RAM rise across a long chain even when final streaming itself remains bounded.

The workflow JSON does not select a cache policy. `--cache-none` is useful as a diagnostic because it disables result reuse and lowers RAM/VRAM use, but it also recomputes every node on each manual run; it is not required for the Update-6 workflows. Bounded or RAM-pressure cache choices remain a ComfyUI runtime decision.

---

## 14. Historical Update-5 checkpoint format

Update 5 long-form workflows saved each completed H3 joint AV sampler output using:

```text
H3 Checkpoint Save
```

The checkpoint is a safetensors file containing at least:

```text
video
audio
```

for the H3 video/audio latent streams.

Fixed clip slots use names like:

```text
clip_00001.safetensors
clip_00002.safetensors
...
```

The save is designed to be atomic so a failed/interrupted write does not silently replace a valid completed checkpoint with a partial file.

### Why save latents rather than intermediate MP4s?

Latents:

- avoid lossy intermediate video encoding;
- are much smaller than decoded full-resolution frame batches;
- preserve the expensive sampler result;
- can be decoded again for final assembly;
- can provide continuation context after restarting ComfyUI.

---

## 15. Historical Update-5 lazy resume

The Update-5 architecture supported resume at **completed clip boundaries**.

The repository does not attempt to resume a sampler halfway through its diffusion steps.

### Music-video resume

`H3 Resume / Live Tail Frames` has a lazy live input.

Normal run:

```text
use live previous clip
```

Resume run:

```text
load previous completed checkpoint
-> decode only the visual information required for the next continuation
-> do not request the earlier live generation branch
```

Because the live input is lazy, ComfyUI can skip the earlier upstream clip tree when the checkpoint path is selected.

### AV-extension resume

`H3 Resume / Live AV Latent` performs the equivalent job for workflows that want the previous **joint H3 AV latent** rather than decoded tail frames.

This allows the multi-clip masked AV extension to continue directly from a saved previous latent.

---

## 16. Historical Update-5 final checkpoint trigger

`H3 Checkpoint Final Trigger` is a lazy dependency helper.

Its job is to make final assembly depend on only the configured last active checkpoint rather than eagerly requesting every optional clip group.

This is especially important in workflows with many available slots but only some active clips.

---

## 17. Historical Update-5 music-video checkpoint assembly

`H3 Assemble Checkpoints` assembled the Update-5 master-song music video from saved H3 checkpoints.

It deliberately does not create one final ComfyUI `IMAGE` output containing the complete movie.

The process is approximately:

```text
load checkpoint 1
-> decode clip 1
-> write completed frames
-> retain only overlap tail
-> release clip 1

load checkpoint 2
-> decode clip 2
-> blend previous tail with current overlap
-> write completed frames
-> retain only next overlap tail
-> release clip 2

...
```

### Linear blend

The implementation matches the source-side KJNodes `linear_blend` convention used by the previous workflow.

For an overlap of `N` frames, alpha values are the interior values of:

```python
linspace(0, 1, N + 2)[1:-1]
```

and each overlap frame is:

```text
(1 - alpha) * previous_tail + alpha * current_overlap
```

The endpoints therefore do not use exact alpha 0 or 1 inside the blended overlap.

Every adjacent clip pair is still blended. For 20 clips there can be 19 seams.

### Output streaming

Decoded images remain floating-point while the seam is calculated. Completed frames are then quantized to RGB24 at the final ffmpeg streaming boundary.

The final H.264 encode happens once.

For the master-song workflow, ffmpeg muxes the original master audio.

---

## 18. Historical Update-5 existing-video checkpoint assembly

`H3 Assemble Extension Checkpoints` performs the corresponding job for normal audiovisual extension.

Update 5 additionally introduced `H3 Start Masked Context`, `H3 Start Canvas Selector`, and `H3 Assemble Starter + Extension Checkpoints` so the same multi-clip masked-AV workflow could either begin from an uploaded source video or from a generated starter clip (pure T2V or I2V). The starter path was also checkpointed and assembled sequentially, so it kept the same low-memory / low-OOM behavior as the source-video path.

Unlike the music-video workflow, generated H3 audio is part of the delivered continuation.

The assembler therefore:

- starts with the original source video/audio;
- loads generated checkpoints sequentially;
- decodes video one clip at a time;
- blends every visual seam;
- decodes generated audio;
- removes the duplicated protected AV prefix from the delivered continuation audio;
- fits audio segments to the exact frame-derived sample timeline so small latent-grid rounding differences do not accumulate;
- streams the final result to ffmpeg.

---

## 19. Historical checkpoint-memory rationale

A decoded ComfyUI `IMAGE` batch is normally a floating-point tensor.

A long sequence at high resolution can therefore consume many gigabytes even before additional intermediate tensors are considered.

With cumulative overlap nodes, multiple increasingly large intermediate batches can coexist because they are graph outputs/cacheable values.

Sequential checkpoint assembly changes the memory shape from approximately:

```text
all clips + all cumulative intermediate movies
```

to:

```text
one decoded clip + one small overlap tail + encoder/runtime overhead
```

Disk becomes the durable intermediate store, while RAM remains bounded by the current clip rather than total movie length.

---

## 20. Runtime compatibility layers

The repository has historically supported ComfyUI versions at different stages of H3 feature development.

The guiding rule now is:

> Use native ComfyUI behavior when the required capability exists. Install compatibility behavior only for the specific missing capability needed by the executed node.

### Native guide compatibility

`h3_compat.py` covers the classic Motion Context/native-guide requirements.

The current capability check verifies live behavior rather than depending on one exact source-code spelling.

It also checks only capabilities relevant to the current conditioning:

- simple Motion Context without refs does not require Ref2VA merge behavior;
- video/image refs require the video merge behavior;
- audio refs require the audio merge behavior.

This avoids false negatives such as the environment reported in repository Issue #7.

### AV-mask compatibility

The repository's H3 masking work was also contributed upstream to ComfyUI in **[PR #15375](https://github.com/Comfy-Org/ComfyUI/pull/15375), “Support per-token video and audio latent noise masks on MiniMax-H3”** (authored as `drozbay`). That upstream PR is the core contribution behind native per-token H3 video/audio denoise-mask handling: it carries the mask payload into the model, derives mask-dependent row timesteps, and applies MiniMax-H3-specific latent inpainting behavior. The compatibility modules in this repository are therefore capability fallbacks for older/intermediate ComfyUI revisions, not a competing permanent mask implementation.

`h3_mask_compat.py` and `h3_mask_payload_compat.py` cover the target-latent video/audio denoise-mask path.

The mask layer is kept separate from normal Motion Context so using a legacy guide workflow does not unnecessarily install masked-target compatibility behavior.

Detection is capability-oriented so newer native ComfyUI support can make the fallback path retire itself.

PR #15375 changed its internal integration on 2026-08-15 (commit `989e7a9`): the earlier `MiniMaxH3.process_denoise_mask` preprocessing hook was removed, and token-grid mask/blend alignment moved into `MiniMaxH3.scale_latent_inpaint(..., x=..., denoise_mask=...)`. The compatibility layer recognizes both layouts. On a ComfyUI build with the newer native architecture it does **not** reinstall the older preprocessing hook; native mask alignment and payload handling remain authoritative.

The workflow-facing mask contract is unchanged: `0 = preserve`, `1 = generate`, with fractional audio values allowed for the AV feather. ComfyUI PR #15375 quantizes fractional mask strengths to a bounded set of levels before deriving per-row timesteps, so the Update-6 half-cosine audio feather remains compatible with the newer native implementation.

### Update 8 step 1: fractional V2V precision — 2026-08-30

`H3 V2V Granular Fractional Denoise` (`H3V2VGranularFractionalDenoise`) encodes the full selected source video/audio interval into the target H3 AV latent and drives V2V strength through H3 denoise masks. **Granular** refers to fractional `0..1` preserve-to-generate mask levels rather than a binary mask. The compatibility path uses ceiling quantization on a 1/4096 grid; it is not infinitely continuous. `0.9995 → 0.99951171875`, `0.9997 → 0.999755859375`, and `0.9998 → 1.0`. Values above 4095/4096 enter the fully-generative top bin. Native support may be finer. **Fractional Denoise** is the user-facing effect of those levels. The node does not replace mask logic for other model families: its near-1 precision compatibility is lazy and capability-probed, and only supplies missing H3 precision/transport behavior when the node executes. It does not modify scheduler sigmas; `BasicScheduler` denoise remains `1.0`. The default global video mask is `0.9995`, while audio defaults to `0.0` for source-audio preservation.

The near-1 precision layer lives in `h3_mask_precision.py` and is deliberately not imported by normal repository startup. The V2V node imports it inside `prepare()` and probes the live H3 implementation before changing anything. The installed H3 class/function changes are process-wide for H3, not local to the returned MODEL instance; they disappear on restart. The capabilities are independent: token-grid resolution/near-1 condition retention, exact-1 row shortcuts, and FP32 transport for `denoise_mask` / `audio_denoise_mask`. A native implementation can therefore replace these pieces incrementally; only still-missing pieces are patched. Once all probes pass natively, the precision layer self-retires and installs nothing.

The precision probe tests more than dtype. It verifies multiple near-1 values so a coarser `1/2048` grid cannot accidentally look equivalent at `0.9995`, and it verifies that `0.99951171875` reaches the H3 diffusion call unchanged as FP32 rather than being rounded in BF16 and merely converted back to FP32 afterward.

The older Update-2/6 AV-mask compatibility remains separate and lazy. Its payload detector recognizes the current merged `_denoise_mask_conds` helper architecture, so current native ComfyUI mask support does not trigger the legacy payload wrapper.

### Shared exact H3 audio-grid preparation

`h3_audio_grid.py` centralizes the repo's source-PCM -> H3-audio-latent contract. It discovers the live audio VAE sample rate and samples-per-latent geometry, requires a PCM length of exactly `target_audio_steps * samples_per_latent`, and verifies that the encode returns exactly `target_audio_steps`. This is not a monkeypatch: it makes the node input conform to the H3 grid before ComfyUI's generic VAE preprocessing can alter it. On the current 32 kHz H3 audio VAE, that geometry is 800 PCM samples per latent cell (25 ms / 40 Hz), so exact-grid preparation also prevents the generic wrapper's centered down-crop from shifting the semantic start or seam.

Affected encoders use different alignment anchors according to their semantics:

- V2V and master-song clips: start/timeline origin is authoritative;
- Motion Context tail audio: continuation seam/end is authoritative;
- Masked AV Bridge: each generated-middle seam is authoritative on its adjacent source side.

`H3 Existing Video Masked Context` restricts the **preserved duration** to shared AV boundaries, so the source-audio span itself is naturally exact-grid and still routes through the shared strict encoder. Update 7 interior inserts from Reithan's PR #3 may place that exact-grid span at a non-joint boundary such as frame 17; only the destination audio-step start is then rounded to H3's 40 Hz grid (with the existing warning). Multiples of 51 align both clocks exactly. `H3 Assemble Interior Insert` restores the source segment at exact frame-derived PCM boundaries after decode, so latent-grid placement quantization does not become final-output seam drift. Latent-copy and decode-only paths never enter the generic audio VAE encoder and are unaffected.

Keep the three audio contracts distinct when debugging:

1. **Before audio-VAE encode:** build an exact H3-grid PCM window with the correct start/end/seam anchor; an encode that returns the wrong latent-step count is an error.
2. **After audio-VAE decode:** tiny frame-vs-40-Hz duration differences may be time-conformed to the exact picture timeline before trimming/splicing.
3. **Final delivery/mux assembly:** public waveforms may still be trimmed or endpoint-padded to an explicit frame-derived delivery length.

These post-decode/output operations do not compensate for a bad PCM -> latent encode.

Stock `MiniMaxH3ReferenceToVideo` audio references still have their own semantic window selection. In Update 9, the current workflows route their audio VAE through `H3 Audio VAE Compatibility` before stock references. The node applies PR #15972-equivalent `crop_input=False` only on an unfixed connected MiniMaxH3AudioVAE wrapper. Native fixed/backported VAEs are unchanged. Repo-owned `encode_exact_audio_grid()` calls make the same lazy check automatically; they still prepare exact PCM spans, because correct timeline anchoring is separate from generic cropping.

### Update 9: lazy compatibility and retirement

Both upstream PRs were open when Update 9 was prepared; this is provenance, not a runtime dependency on GitHub. Installed capabilities determine whether a fix is needed. A merge does not fix an old local installation until the user updates it, and a backport can provide the fix before a merge.

- **PR #15972:** narrow VAE-instance configuration correction, checked on node use. No global VAE constructor replacement; unrelated VAEs are unchanged. A fixed native instance receives no patch. The 437,333-sample regression confirms origin preservation and 547 cells instead of the cropped 546.
- **PR #15988:** live-forward behavioral probes use small synthetic network outputs with fractional video/audio masks, including audio carry scale 4. An unfixed recognized forward receives the upstream velocity multiplication after the network result/output-copy stage but before audio carry conversion. Native passing implementations are untouched. Unknown or partially converted layouts raise rather than risk double scaling; post-install failure rolls back the function.
- **Fractional precision:** the audio shortcut probe evaluates the extracted live audio-mask branch with near-one masks, rather than assuming an absent literal cutoff string means native readiness. Source/AST availability is still required for this branch check and compatibility rewrites. Known patched functions retain their explicit marker.

PR #15988 scales predicted velocities by the denoise mask before global-sigma x0 recovery, so masked rows evaluated at `mask * sigma` are converted consistently. This applies to ordinary feathered masks as well as near-one V2V masks. Attribution: poorpaper’s [ComfyUI PR #15988](https://github.com/Comfy-Org/ComfyUI/pull/15988).

### Update 9: other runtime repairs

Spatial V2V masks now use the same center-crop rectangle as source frames before latent resizing. Shortened Motion Context audio uses its actual returned duration for seam-nearest placement; complete windows retain the old coordinate and saved widget order. Current workflow clocks are fixed at 24 fps while old Python source-normalization helpers remain available for legacy integrations.

The AV/Music streamer resolves `%date:yyyy-MM-dd%/MiniMax_H3_` and documented time tokens using the server’s local clock, including API queues. Prefixes remain relative to ComfyUI output; traversal/absolute paths are rejected. VHS owns output numbering. This addresses [issue #16](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/issues/16).

---

## 21. Main node reference

### Guide/legacy nodes

| Display name | Purpose |
|---|---|
| H3 Motion Context | Native H3 previous-motion/audio guide conditioning |
| H3 Motion Context Trim | Remove repeated guide prefix and produce delivery overlap |
| H3 Motion Context Save Latent | Older Motion Context latent persistence helper |
| H3 Motion Context Load Latent | Older Motion Context latent load helper |
| H3 Custom Keyframes | Native H3 still-image anchors at chosen timeline positions |

### Latent-masking nodes

| Display name | Purpose |
|---|---|
| H3 Existing Video Masked Context | Start masked continuation from ordinary decoded source AV, at an arbitrary `insert_frame` |
| H3 Custom Keyframes (Masked) | Hard-preserve still-image keyframes via latent write + noise mask |
| H3 Generated AV Masked Context | Continue from a previous generated H3 AV latent directly |
| H3 Masked AV Bridge | Protect source A/B endpoints and generate the middle |
| H3 Song Audio + Masked Video Context | Put the master-song interval into target audio latent and optionally protect previous visual context |
| H3 Optional Reference Image | Lazy optional global reference-image slot |
| H3 Crop Source To /32 | Prepare source image geometry for H3 workflows |
| H3 V2V Granular Fractional Denoise | Encode full source AV into the target H3 latent and attach H3 fractional denoise masks with lazy/self-retiring precision compatibility |

### Direct-latent long-form nodes

| Display name | Purpose |
|---|---|
| H3 Start Masked Context | First-extension selector: source-video prefix or generated-starter latent tail |
| H3 Start Canvas Selector | Shared source/generated resolution selector |
| H3 Stream Final AV Extension to VHS | Decode source/starter + 1–6 live extensions sequentially and stream completed frames directly into VHS without a full final IMAGE movie tensor |
| H3 Song Audio + Masked Video Context | Insert the exact master-song slice and optionally copy the previous generated video latent tail |
| H3 Stream Final Music Video to VHS | Decode up to 20 live clip latents sequentially and stream directly into VHS while passing the untouched master song |
| H3 Final Stream Output Sink | Terminal `VHS_FILENAMES` sink used by Music Video to keep clip-preview scheduling ahead of the all-clips final-stream dependency chain |
| H3 AV Extension Controller | Start mode, active extension count, audio feather, and preview policy |
| H3 Music Video Controller | Active clip count and preview policy |
| H3 Assemble Existing Video Extension | Single-extension source + continuation assembly helper |
| H3 Assemble Interior Insert | Required pixel-correct output path for interior inserts |

The Update-5 checkpoint/resume nodes are no longer registered in Update 6. The historical sections above remain only to document the previous implementation and migration context.

---

## 22. Workflow categories

The current highlighted workflows use the `NEW -` prefix:

### `NEW - 2MP De-Rope Continuation - Working Example.json`

Two-pass de-rope continuation with a hold-map-derived interior Motion Context seam guide and no pass-2 denoise mask.

### `NEW - AV Extension.json`

General long-form extension from an existing video or generated T2V/I2V starter, with optional global image/audio references.

### `NEW - Music Video.json`

Long-form song-driven generation with exact master-song slices and direct previous-latent continuation.

### `NEW - V2V Latent Motion Transfer (with upscale and de-rope).json`

Update-8 two-pass V2V refinement: original source motion anchors the first fractional-denoise V2V pass; de-rope is derived only after the first denoised/x0 estimate; the generated latent is then learned-upscaled and refined in a second pass before exact timeline recovery.

Secondary examples are clearly separated:

- `UTILITY - AV Bridge.json`
- `UTILITY - Custom Keyframes.json`
- `OLD - Motion Context - Simple.json`
- `OLD - Motion Context - Advanced.json`
- `OLD - Hybrid Extension.json`

Direct latent continuation is an implementation detail of AV Extension and Music Video, not a separate workflow category. The V2V latent-motion-transfer example is a separate two-pass refinement design.

---

## 23. Testing

The repository contains CPU/static/mock regression tests because full H3 inference requires the user's ComfyUI model/runtime installation.

The tests cover, among other things:

- native Motion Context structure;
- Simple and Advanced legacy workflow wiring;
- native guide/reference compatibility detection;
- per-token AV mask capabilities;
- existing-video extension behavior;
- direct generated-AV latent continuation;
- master-song audio masking;
- the one-token audio-grid boundary regression;
- direct-stream final workflow wiring and absence of a separate full-movie VHS input chain;
- absence of the superseded materialized full-movie Update-6 assembler nodes;
- Music Video intermediate-stream/terminal-sink scheduling topology;
- same-result VHS inline-preview refresh behavior;
- exact decoded-audio timebase conformance and no-silence-tail regression;
- streamed-frame equivalence against the former full-buffer seam math;
- one-shot VHS frame-sequence behavior;
- workflow JSON consistency;
- dynamic de-rope seam fan/offset wiring, hold-map validation, and absence of a pass-2 denoise mask;
- `insert_frame` offsets: zero is byte-identical to the prefix path, non-multiples of 17 snap down to the grid, video/audio step ranges for inserts at 17, 51, 102 match hand-computed values;
- `H3 Assemble Interior Insert`: frame count unchanged, audio samples match, splice interval equals canonical source, same CFR index map as the context node;
- `H3 Custom Keyframes (Masked)`: frame-to-step mapping across all five phases, duplicate quantized steps raise, nested H3 AV masks merge on non-overlapping steps, protected-step overlaps raise, plain single-tensor masks are rejected, video mask zeros exactly at pinned steps, and upstream audio mask state is preserved;
- the JS keyframe widget covering both keyframe node names.

Run the standalone repository regressions with:

```bash
pytest -q
```

Install `requirements-test.txt` first. The pytest entry point delegates to `tests/run_tests.py`, which runs each module in a fresh pytest subprocess. Pytest fixtures execute normally and incompatible ComfyUI mocks cannot leak between modules. The same checks run directly with `python tests/run_tests.py`. CI explicitly installs pytest. Update 9 includes behavior regressions for memory lifetime, spatial cropping, short audio placement, absolute bridge samples, date prefixes, and native/legacy compatibility retirement.

---

## 24. Additional focused references

- [MODIFICATIONS.md](MODIFICATIONS.md) — release history.
- [example_workflows/README.md](example_workflows/README.md) — practical usage notes for the included workflows.
