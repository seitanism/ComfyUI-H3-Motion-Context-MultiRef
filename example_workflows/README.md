# Example Workflows

Update 9 examples require constant **24 fps** input videos. See [per-workflow prerequisites](../WORKFLOW_PREREQUISITES.md) for the exact external packs, recorded versions, model filenames, and assets. Generation/output clocks are fixed at 24; old widget names remain for saved-workflow compatibility.


## NEW - 2MP De-Rope Continuation — Working Example

`NEW - 2MP De-Rope Continuation - Working Example.json`

A focused Clip-1 + Extension-1 example for two-pass MiniMax H3 de-rope at 2MP. The continuation uses MAINodes `Jerk Oracle`, `Time Smear`, `Audio Smear`, and `Exact Recover`, plus the H3 latent upscaler.

This is intentionally a **working example rather than a tidy showcase workflow**; the underlying seam, fan, timing, and two-pass plumbing is left visible for inspection.

The seam path is deliberately mask-free at pass 2:

1. take the recovered final 39 frames of Clip 1;
2. fan them with the exact `hold_map_used` from Extension 1;
3. overwrite the beginning of the low-resolution smeared pass-2 init;
4. return the seam-nearest 39 fanned frames at native Clip-1 resolution plus their dynamic target offset;
5. use those frames as one native H3 Motion Context clip at that interior offset;
6. run the 2MP refinement pass and `Exact Recover`;
7. trim the duplicated real-time prefix from the first continuation stage and assemble with the existing exact AV extension nodes.

The guide offset is derived from the hold map at runtime; it is not a fixed frame number. This is conditioning placement, not latent preservation: `target_start` on `H3 Motion Context` is separate from Update 7 (2026-08-18), which introduced `insert_frame` on `H3 Existing Video Masked Context`.

Required external packs for this example include ComfyUI-MAINodes, ComfyUI-KJNodes, ComfyUI-SolAttn_triton, and Comfyui_Minimax_h3_latent_Upscaler.

The bundled example uses `assets/derope_continuation_reference.png`; copy that file into ComfyUI's `input/` folder before running the example, or replace the Load Image node with your own reference.

---

## NEW - V2V Latent Motion Transfer (with upscale and de-rope)

`NEW - V2V Latent Motion Transfer (with upscale and de-rope).json`

Transfers an existing source video's motion and timing into a new H3 generation, then refines it with de-rope and learned latent upscaling. This workflow is meant for **motion transfer rather than source-video restyling**: the source clip provides motion, pose, camera movement, and timing guidance, while the generation can replace the subject's appearance and identity through the normal H3 reference path.

### What the workflow does

The workflow runs in two passes:

1. **Pass 1: latent motion transfer.** The original source video/audio is used directly as the V2V and Ref2VA anchor. A first H3 generation is produced while preserving just enough source-latent structure to carry over motion and timing.
2. **Pass 2: upscale and de-rope refinement.** The pass-1 result is de-roped from generated x0, time-smeared for the second pass, upscaled in latent space, and refined again at the higher resolution.
3. **Final recovery.** `H3ExactRecover` restores the original 24-fps real-time timeline after de-rope processing, so the final clip preserves the intended playback speed and duration.

### Why the fractional mask matters

Keep the V2V stage's `BasicScheduler` denoise at `1.0`; V2V strength is controlled by the **H3 V2V Granular Fractional Denoise** node instead. The important control is `global_strength`, which sets a near-1 denoise mask over the source latent.

This is important because standard denoise handling would force a binary choice:

- preserve too much of the source latent and the result stays too close to the original clip; or
- generate too freely and the motion/performance guidance weakens or collapses.

The granular fractional path lets the workflow keep only a **faint residue** of the source latent inside the target latent. That faint residue is often enough to stabilize motion transfer, pose continuity, and timing while still allowing the subject and look to change.

### Recommended `global_strength` range

For this workflow, the practical range is usually **`0.996` to `0.9997`**. That narrow range is unusually important:

- **too low** (for example clearly below `0.996`) keeps too much of the source video latent, which can make the output cling to the original subject or composition;
- **too high** (approaching or effectively behaving like `1.0`) removes too much of the helpful source-latent residue and can weaken the transferred motion guidance;
- **around `0.9995`** is often a strong starting point, because it still leaves a tiny amount of source residue while giving the model enough freedom to replace identity and appearance.

If the source motion is **not transferring strongly enough**, decrease `global_strength` in small steps. A lower value preserves a little more of the source latent, which strengthens the motion/performance guide. Keep the changes small and tune against the specific source clip; the useful range is intentionally narrow.

The token grid is finite: the compatibility patch uses ceiling quantization at **1/4096**. For example, `0.9995 → 0.99951171875`, `0.9997 → 0.999755859375`, and `0.9998 → 1.0`. Small slider changes can share a grid value; native implementations may offer finer precision. The recommended range is empirical tuning guidance, not a guarantee for every model/source.

Update 8's granular fractional-denoise support is what makes this usable: it carries the H3 denoise-mask values in FP32 and preserves near-1 values instead of collapsing them to plain `1.0`. Without that precision, settings such as `0.9995` would lose their intended meaning.

### Practical workflow notes

- The workflow already includes the second-pass de-rope and latent upscale stages; you do not need a separate refinement workflow after it.
- Streamed blocks are kept intentionally because this workflow is memory-heavy; removing them can easily push large runs into OOM territory even on a 5090, especially once de-rope multiplies the internal frame count.

The granular fractional V2V node (`H3V2VGranularFractionalDenoise`, displayed as `H3 V2V Granular Fractional Denoise`) is supplied by this MultiRef repository in Update 8 (started 2026-08-30).

This workflow additionally uses ComfyUI-MAINodes and the LBH-123-AI MiniMax H3 latent upscaler custom node.

---

## NEW - Music Video

`NEW - Music Video.json`

Creates a multi-clip music video around one song.

The included workflow comes with a complete example using the files in:

`example_workflows/assets/`

Copy the example images and song into your ComfyUI `input/` folder before running it.

### Main controls

**Active Clips**
Sets how many clip sections are used.

**Previews**
Controls which clip previews are generated.

**Reference Images**
Use the included references or replace them with your own.

**Master Song**
The song used by the workflow. Replace it with your own audio when starting a new project.

The workflow contains up to 20 clip sections. Only the number selected by **Active Clips** is used. Changing that count does not alter existing clip-sampler inputs, so cached clips can be reused while newly activated continuation clips are generated. The final streamer and last-active preview barrier both support a standalone dynamic **Input Count / Update inputs** interface; Music Video clip connections must remain a contiguous prefix so their visuals stay aligned to the master-song timeline.

---

## NEW - AV Extension

`NEW - AV Extension.json`

Continues a video across multiple H3 generations.

It can start from either:

- an existing video;
- a new T2V generation;
- a new I2V generation.

### Start mode

Choose whether the workflow begins with an existing video or generates the first clip itself. Existing Video uses the normal VHS loader so the source clip has an inline preview. Its VHS audio output is deliberately unconnected; **Keep source audio** extracts audio safely inside the MultiRef node and treats a genuinely silent container as silence, while **Regenerate with H3** protects the full source picture and synthesizes a replacement soundtrack.

Keep source audio does not regenerate the whole context. With 39 frames and eight feather ticks, the first 57 audio ticks remain protected and only the final 8 ticks (0.2 s) gradually denoise. Assembly uses the extension decode over the overlap, so VAE reconstruction can still differ slightly.

Final AV/Music prefixes accept `%date:yyyy-MM-dd%/MiniMax_H3_` and relative subfolders; dates use the server clock.

### Extensions

Set **Active Extensions** to the number of continuation sections you want to run. The bundled controller automatically enables the required managed extension groups and bypasses the inactive ones; no manual group-enabling order is required.

The final **H3 Stream AV Extensions to VHS** node can also be reused in custom workflows. Set **Input Count**, click **Update inputs**, and connect any number of extension latents up to the configured count. Disconnected sockets are skipped. The optional `active_extensions` input is only a cap; leave it disconnected when using the node outside the bundled controller workflow. The bundled graph also routes the enabled per-extension VHS previews through a last-active preview barrier. The barrier has its own **Input Count / Update inputs** control, so only the preview sockets you need are shown; the highest enabled preview completes before the final stream starts. Its terminal sink is optional-input/bypass-safe.

### References

Optional reference images can be used when needed.

### Previews

Use the preview control to enable or disable extension previews.

---


## UTILITY - AV Bridge

`UTILITY - AV Bridge.json`

Use two constant-24-fps videos with audio. **H3 AV Bridge Timing** validates the target and context before generation:

- context length: **39, 90, 141, 192, …** (`39 + 51*k`); 56 is not allowed;
- total target length: **5 + 17*k**;
- the target must be longer than both preserved contexts combined.

Defaults **192 target / 39 preserved per side** give **1.625 s** protected at each side and a **4.75 s** generated middle. H3 Assemble Bridge Audio trims the protected audio intervals on the 40 Hz grid and conforms the generated middle to exact frame-derived sample boundaries before concatenating both sources. This also handles valid targets such as 107 or 124 whose audio endpoint is rounded to a latent cell. Keep both visual overlap links connected to the bridge's validated preserve_frames output.

The example requires VideoHelperSuite and KJNodes; see [its prerequisites](../WORKFLOW_PREREQUISITES.md).

---

## UTILITY - Custom Keyframes

`UTILITY - Custom Keyframes.json`

This example demonstrates soft image guides. Read [Keyframes, masks, and interior inserts](../KEYFRAMES_AND_INSERTS.md) for detailed soft/hard keyframe instructions, one-based versus zero-based positions, latent-step quantization, Update 7 interior-insert recipes, and H3 Set/Clear AV Noise Mask usage. The three `OLD -` workflows remain available for earlier setups.


## More information

For implementation details, timing, masking, audio handling, and other internals, see:

[../TECHNICAL_ARCHITECTURE.md](../TECHNICAL_ARCHITECTURE.md)

For the detailed update history, see:

[../MODIFICATIONS.md](../MODIFICATIONS.md)
