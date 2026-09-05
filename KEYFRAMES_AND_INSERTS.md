# Keyframes, AV masks, and interior inserts

This guide covers the custom keyframe utilities and related preservation nodes, especially the Update 7 features contributed by **Reithan** in [PR #3](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/3). Update 9 keeps their public node names and saved widget order and improves compatibility, timing guidance, and shortened timeline-audio placement.

Start with `example_workflows/UTILITY - Custom Keyframes.json` for **soft image guides**. That example does not demonstrate every node below. See [workflow prerequisites](WORKFLOW_PREREQUISITES.md) for model files. These nodes use H3 video/audio VAEs and H3 AV latents, not generic image-model latents.

## Choose the right tool

| Node | What changes | Use it when |
| --- | --- | --- |
| H3 Custom Keyframes | Conditioning guides | You want H3 to follow still images at selected points while generating a coherent clip. |
| H3 Custom Keyframes (Masked) | Video latent values and video noise mask | You need encoded still anchors protected from denoising. |
| H3 Existing Video Masked Context | Video/audio latent content and masks | You have decoded source frames/audio and want to preserve their tail inside a new target. |
| H3 Assemble Interior Insert | Decoded pixels and PCM audio | You need the source inserted back over an interior latent-preserved region. |
| H3 Set AV Noise Mask | Video/audio denoise masks | You need to edit stream-specific preserve/generate regions without replacing latent samples. |
| H3 Clear AV Noise Mask | Removes the complete noise mask | You want to discard prior protection before deliberately rebuilding it. |
| H3 Generated AV Masked Context | Copies a generated AV latent tail | You want to continue a generated clip without decoding and re-encoding its context. |
| H3 Motion Context | Native video/audio guide conditioning | You want a motion guide, including one placed at a de-rope interior seam. |
| H3 Fan Recovered Context | Smeared baseline and guide placement | You need to map a recovered seam onto a Time Smear hold map. |

A noise mask is **0 = preserve, 1 = generate**. Fractional values allow partial denoising. Removing protection means setting a stream to **one**, not zero.

## Frame indexing and H3 grids

All current examples operate at **24 fps**. A full target video uses `5 + 17*k` frames: 5, 22, 39, 56, 73, 90, … . Video-latent steps cover a repeating `1, 4, 4, 4, 4` frames.

Do not confuse these different grids:

| Control | Valid/preferred positions or lengths |
| --- | --- |
| Full target frame count | `5 + 17*k` |
| AV continuation/bridge context length | `39 + 51*k`: 39, 90, 141, 192, … |
| Still-token positions, zero-based | `17*k`: 0, 17, 34, 51, … |
| Still-token positions, one-based | `1 + 17*k`: 1, 18, 35, 52, … |
| Existing-video insert offset | Multiples of 17; multiples of 51 also align audio exactly |

The two custom-keyframe nodes default to **one-based** positions. Their `indexing` control changes interpretation; changing it does not automatically shift the values you entered. The last valid position is the target frame count in one-based mode, or one less in zero-based mode.

## H3 Custom Keyframes — soft guides

Inputs: `conditioning`, video `vae`, target `latent`, `indexing`, `crop`, and dynamically added image/position pairs. Output: `CONDITIONING`.

1. Connect the H3 conditioning and matching target latent from the normal H3 source node.
2. Connect the H3 video VAE.
3. Attach exactly one image to each visible keyframe slot.
4. Set each position; use **+ Add keyframe** or **- Remove keyframe** for 1–32 slots.
5. Feed the output conditioning into your guider/sampler, and feed the original target latent into the sampler.

Position sockets can override the corresponding position widgets. The internal `keyframe_state` string stores widget state; normally leave it to the frontend. Duplicate positions, missing images, image batches, and positions outside the target raise errors.

The node resizes/crops each image and encodes a still guide. It replaces this conditioning's `minimax_keyframes` list with the selected custom anchors; it does not hard-write those anchors into the sampler latent or create a denoise mask. Use this when a suggested pose/composition is more useful than strict latent protection. Guidance does not guarantee exact image reproduction.

## H3 Custom Keyframes (Masked) — protected still latents

Inputs: target H3 `latent`, video `vae`, and the same dynamic keyframe UI. Output: `LATENT`.

Feed this node's output into the sampler's latent input. Ordinary text/reference conditioning remains connected through the guider as usual.

The node encodes each still, writes it into the target's containing video-latent step, and sets that step's video mask to zero. For one-frame step coverage, choose the still-token positions above. An interior position belongs to a four-frame step, so protection applies to the whole containing step. This can produce hold-like behavior, but **protecting a still latent is not a guarantee of identical decoded pixels or an exactly static four-frame output**. The causal VAE uses surrounding latent context.

If two requested positions quantize to the same latent step, the node rejects them. If any part of a step is already protected by an incoming video mask, the node rejects a new still anchor there rather than overwrite another preservation source.

An incoming nested H3 AV mask is retained: new keyframe protection is added only to otherwise unprotected video steps, and the **audio mask remains unchanged**. Without an incoming audio mask, audio defaults to all-generate. The node accepts the two-stream H3 mask contract; use H3 Set/Clear AV Noise Mask to construct/reset masks instead of feeding an arbitrary plain tensor.

## H3 Existing Video Masked Context — preserve a source interval

Inputs include the target H3 latent, both VAEs, source frames/audio, `context_length`, `crop`, `audio_feather_ticks`, and `insert_frame`. Current workflows require constant-24-fps input; retained older Python calls can still normalize explicitly declared source frame rates.

The node selects the source's tail, snaps the requested context down to a fitting shared AV boundary, encodes it, and writes it inside the target with preservation masks. Prefer explicitly selecting 39, 90, 141, 192, … instead of relying on snapping. The target must leave generated content outside the preserved span.

`insert_frame` is zero-based. Nonmultiples of 17 snap downward. A 51-frame offset is both video-phase and audio-grid aligned; a 17-frame offset is video-phase aligned but requires approximately 8.33 ms of destination audio-cell rounding. The source audio window itself still has an exact-grid duration.

Outputs:

| Output | Meaning |
| --- | --- |
| `latent` | Prepared samples plus nested video/audio mask |
| `trim_frames` | Prefix overlap count for the head-extension path |
| `insert_frame` | Actual snapped destination position |
| `preserved_frames` | Actual source span written into the target |

At the default 39-frame / 65-audio-tick context with eight feather ticks, the first **57 audio ticks remain fully protected**; only the final **8 ticks (0.2 s)** follow the half-cosine denoise release. The video context remains protected. Set `audio_feather_ticks=0` when fully protected audio is needed throughout the interval. The default AV final streamer deliberately uses the extension's decoded overlap so that release is retained; it does not denoise the complete context.

## H3 Assemble Interior Insert — exact source splice after decoding

For `insert_frame > 0`, use this node after sampling and decoding the complete continuation. Do not use the prefix-only trim/append recipe for an interior insert.

Connect:

- the complete decoded generation to `continuation_images` and `continuation_audio`;
- the same original source frames/audio used by the context node;
- the context node's actual `insert_frame` and `preserved_frames` outputs;
- matching crop settings and the fixed 24 fps timeline.

The assembler replaces the protected interval with the canonical source pixels and source PCM. This is a hard splice, not a crossfade, and leaves the total continuation frame count unchanged. “Exact source pixels” means the same resized/cropped source used by preparation, not the original uncropped file or lossless final codec output.

This source splice intentionally replaces any generated audio feather inside that inserted interval. If preserving the feather is the goal, use the ordinary head-extension assembly policy instead. The two operations have different purposes.

### Recipe: interior video plus additional masked keyframes

1. Create a sufficiently long H3 target, for example 192 frames.
2. Apply Existing Video Masked Context with 39-frame context and `insert_frame=51` (protects zero-based frames 51–89).
3. Apply Custom Keyframes (Masked) **after** it, with still-token anchors outside that interval, for example one-based 1 and 137.
4. Sample, decode video/audio, and run Assemble Interior Insert with the context outputs.

Putting the existing-video node after the masked-keyframe node would rebuild masks and samples rather than compose in the intended order. A protected-step overlap is an error; move the still anchor rather than silently overwrite it.

## H3 Set AV Noise Mask / H3 Clear AV Noise Mask

Set accepts `latent` plus optional `video_mask` and/or `audio_mask`. It resizes the supplied mask to each stream's layout and keeps the omitted stream's existing mask; without an existing stream mask it uses all ones. At least one supplied mask is required. Samples themselves are unchanged.

The generic mask conversion resizes a temporal mask to the target latent time axis; it is not a frame-accurate keyframe-position editor. Use Custom Keyframes (Masked) or Existing Video Masked Context when precise H3 frame/phase placement matters. Audio masks are spatially averaged, resampled over the uniform 40 Hz timeline, and broadcast over stereo.

Examples:

- Freeze already-encoded audio while generating video: supply an all-zero audio mask and leave the video mask omitted/all-one.
- Allow audio regeneration while retaining video protection: set the audio mask to all-one and omit the video mask.
- Discard all old masks: use Clear, then deliberately rebuild any protection you still need.

Setting a zero mask on an empty latent does not invent valid source content. Encode/copy the intended source first. Clearing an existing video/audio mask can regenerate everything that was protected; it does not restore an earlier latent value. These nodes use the nested H3 AV contract; stock single-stream mask nodes are not substitutes.

## Related continuation and compatibility nodes

**H3 Generated AV Masked Context** copies a valid generated latent tail into a new target without a decode/encode round trip. Keep source/target geometry consistent and use shared AV context lengths. It supports the same audio feather.

**H3 Motion Context** uses native conditioning rather than hard latent preservation. `target_start=0` places a head guide and returns the duplicated-head trim count. A positive target offset places an interior guide and returns zero trim. In Update 9, if timeline audio is genuinely shorter than requested, placement uses its returned duration so it ends nearest the visual seam. Complete audio windows and saved input ordering remain unchanged; nonintegral frame placement is rounded to the nearest frame.

**H3 Fan Recovered Context** uses the exact hold map from the same H3 Time Smear node to fan the recovered previous tail, repair the pass-2 baseline, and calculate the dynamic guide position. Keep it connected as in the 2MP example; an unrelated hold map gives the wrong seam clock.

**H3 Audio VAE Compatibility** takes and returns the connected H3 audio VAE. It lazily disables the unfixed generic audio center crop (PR #15972 equivalent). Route it before stock H3 audio-reference nodes; repo-owned audio encodes also check automatically. Native fixed VAEs are untouched. Exact-grid PCM preparation remains in place because it also defines the intended timeline, independently of cropping.

Masked nodes lazily check PR #15988-equivalent velocity conversion. The fallback modifies the live H3 class for the process and disappears on restart; native behavior is tested before installing it. No merge-status network request is made during sampling.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing image/position errors | Every active keyframe slot needs one image and an in-range position. Remove unused slots. |
| Duplicate/overlap error | Distinct frame numbers can land in one four-frame latent step; also avoid already protected inserts. |
| Audio unexpectedly regenerates | Check the audio mask and avoid replacing the nested AV mask with a plain stock mask. |
| Source splice looks different | Match crop/resolution; hard latent protection alone is not pixel reconstruction. Use the interior assembler where appropriate. |
| Bridge rejects 56 frames | 56 is a valid full video run, but not a shared AV context. Use 39 + 51*k. |
| Compatibility probe refuses to patch | Update ComfyUI or use a verified revision; unknown runtime layouts are not patched blindly. |

For internals see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md). Update history and original contributions are in [MODIFICATIONS.md](MODIFICATIONS.md).
