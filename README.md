# ComfyUI H3 Motion Context — MultiRef & Latent Masking

MiniMax H3 nodes and workflows for video extension, music videos, motion transfer, and custom keyframes.

**Update 9 — 2026-09-05:** fixes streamed-output memory retention and spatial masks, tightens AV Bridge timing, adds dated output folders, and checks H3 audio/mask compatibility lazily. See [MODIFICATIONS.md](MODIFICATIONS.md) for the complete dated update history and technical release details.

## Choose a workflow

| Workflow | Use it for |
| --- | --- |
| [NEW - AV Extension](example_workflows/NEW%20-%20AV%20Extension.json) | Continue an existing video, or generate a T2V/I2V starter and extend it. |
| [NEW - Music Video](example_workflows/NEW%20-%20Music%20Video.json) | Generate connected clips around one master song, with optional reference images. |
| [NEW - V2V Latent Motion Transfer](example_workflows/NEW%20-%20V2V%20Latent%20Motion%20Transfer%20%28with%20upscale%20and%20de-rope%29.json) | Transfer source motion using fractional H3 V2V denoise, then de-rope, upscale, and refine. |
| [NEW - 2MP De-Rope Continuation — Working Example](example_workflows/NEW%20-%202MP%20De-Rope%20Continuation%20-%20Working%20Example.json) | Inspect a two-pass continuation with dynamic de-rope seam placement. |
| [UTILITY - AV Bridge](example_workflows/UTILITY%20-%20AV%20Bridge.json) | Generate a transition between two known video/audio endpoints. |
| [UTILITY - Custom Keyframes](example_workflows/UTILITY%20-%20Custom%20Keyframes.json) | Place image guides along a generation. See the [keyframes and inserts guide](KEYFRAMES_AND_INSERTS.md) for hard masks and related utilities. |

Use **constant 24 fps input videos**. H3 generation and output run at 24 fps; the current examples validate the source/loaded frame-rate metadata. Convert other frame rates before loading. Metadata alone cannot certify that a container uses constant rather than variable frame timing.

For AV Bridge, preserved contexts must be **39, 90, 141, 192, …** frames (`39 + 51*k`). Target lengths use **`5 + 17*k`** and must exceed both contexts combined. Start with **192 target / 39 context**.

For AV Extension, **Keep source audio** protects the context; only its final **8 audio ticks (0.2 seconds)** gradually receive denoising by default. The complete 1.625-second context is not regenerated. Final assembly uses the extension's decoded overlap, so VAE reconstruction can still differ slightly. Music Video keeps the master song as its final soundtrack.

V2V fractional strength is quantized, not infinitely continuous. The compatibility grid uses **1/4096** steps: `0.9995 → 0.99951171875`, `0.9997 → 0.999755859375`, and `0.9998 → 1.0`. Start around `0.9995`; keep `BasicScheduler` denoise at `1.0`. See the workflow note and [workflow guide](example_workflows/README.md) for tuning.

## Install and run

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef.git
```

Restart ComfyUI and refresh the browser. When testing the Update 9 ZIP, replace the existing pack directory rather than installing a second copy alongside it.

Install the packs and model files listed for your chosen example in [Workflow prerequisites](WORKFLOW_PREREQUISITES.md). Models are not included. The examples use different external packs; you do not need every pack for every workflow.

The AV/Music controllers select active sections and previews. Their final streamers support **Input Count → Update inputs**. Existing sampler results remain reusable while retained in ComfyUI's cache; these workflows do not provide disk checkpoint/resume across restarts.

Final AV/Music output prefixes support subfolders, including `%date:yyyy-MM-dd%/MiniMax_H3_`. Dates use the server's local clock. VHS still allocates numbered filenames inside ComfyUI's output directory.

## Guides

- [Workflow guide](example_workflows/README.md) — controls and usage.
- [Workflow prerequisites](WORKFLOW_PREREQUISITES.md) — packs, recorded versions, model filenames, and assets for each example.
- [Keyframes, masks, and interior inserts](KEYFRAMES_AND_INSERTS.md) — detailed node recipes and limits.
- [Technical architecture](TECHNICAL_ARCHITECTURE.md) — timing and implementation contracts.
- [Modifications](MODIFICATIONS.md) — Update 9 details and earlier releases.

## Credits and license

Modified fork of [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context). Update 7's arbitrary inserts, masked keyframes, and AV mask utilities were contributed by **Reithan** in [PR #3](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/3). Update 9's velocity compatibility follows **poorpaper's ComfyUI PR #15988**. See the changelog for attribution.

GPL-3.0. See [LICENSE](LICENSE).
