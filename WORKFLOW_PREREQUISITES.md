# Workflow prerequisites — Update 9

All current examples require this Update 9 pack and a ComfyUI build with MiniMax H3 nodes. Use constant 24 fps sources and H3-compatible model/quantization support. The Python runtime must provide PyTorch, NumPy, safetensors, and torchaudio for audio resampling. VHS output/source audio extraction also needs ffmpeg (system or imageio-ffmpeg). Follow each external pack’s installation instructions for its platform dependencies.

**Version evidence:** versions below come from the supplied workflow metadata. They are reproduction references, not minimum-version guarantees or a claim that every combination was inference-tested for Update 9. CPU regressions were run with Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.2, safetensors 0.8.0, and pytest 9.1.1. GPU inference and the external packs require testing in your ComfyUI installation.

Native guide/MultiRef paths need ComfyUI PR #15439-equivalent behavior; masked paths need #15375-equivalent behavior (the existing fallback remains). Update 9 supplies lazy #15972/#15988 compatibility when the connected VAE/live H3 model lacks the corresponding fix. Native/backported fixes win automatically. There is no GitHub request during generation.

## Model locations

Use the node loaders to select matching files. Standard locations are `ComfyUI/models/diffusion_models/` for UNETLoader, `models/text_encoders/` for CLIPLoader, `models/vae/` for VAELoader, and `models/loras/` for LoRAs. Use the latent-upscaler pack’s documented model location for its weights. Model download/licensing and accelerator requirements depend on the selected model variant; this repository does not bundle or verify third-party model downloads. NVFP4/AWQ and INT8 variants require suitable runtime/hardware support; choose a supported equivalent if needed.

Optional/bypassed LoRA loaders are included in the lists below. Supply their files if enabled, or explicitly bypass/remove the loader. Replacing a turbo LoRA may require changing sampler settings.

## NEW - 2MP De-Rope Continuation - Working Example

File: `example_workflows/NEW - 2MP De-Rope Continuation - Working Example.json`

Core node metadata: `0.30.0`, `0.31.0`, `0.33.0`.

| External pack | Recorded version/revision |
| --- | --- |
| [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | `d7c01b9011f2e8439493f6c02c29995a27df276f` |
| [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `3f20054214fec9f9234fd3841ae6f1e4287948f6`, `version not recorded` |
| [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | `842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc` |
| [matlowai/ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) | `c49cde8826da7460361b408bd590dc86d36cc0b2` |

| Loader | Selected file |
| --- | --- |
| CLIPLoader | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| MinimaxH3LatentUpscaler3D | `minimax_h3_latent_upscaler_3d_fp16.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| UNETLoader | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` |

Selected input assets: `derope_continuation_reference.png`.

Copy `example_workflows/assets/derope_continuation_reference.png` into `ComfyUI/input/`. SolAttn/Triton and the CUDA latent upscaler have platform-specific requirements; this is the most demanding example.

## NEW - AV Extension

File: `example_workflows/NEW - AV Extension.json`

Core node metadata: `0.30.0`, `0.31.0`, `0.33.0`.

| External pack | Recorded version/revision |
| --- | --- |
| [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `1.7.7`, `4ee72c065db22c9d96c2427954dc69e7b908444b` |

| Loader | Selected file |
| --- | --- |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |

Selected input assets: ``.

Existing Video requires a constant-24-fps clip selected in `VHS_LoadVideoFFmpeg`. The VHS AUDIO output remains disconnected; source audio is extracted by H3 Source Audio Policy, including exact-duration silence for a silent container. T2V/I2V modes do not require that source clip.

## NEW - Music Video

File: `example_workflows/NEW - Music Video.json`

Core node metadata: `0.30.0`, `0.31.0`, `0.33.0`.

| External pack | Recorded version/revision |
| --- | --- |
| [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `1.7.7` |

| Loader | Selected file |
| --- | --- |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |

Selected input assets: `I'll Know You by the Scar.wav`, `be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png`, `c90ee577-98eb-4f6c-9b0c-562a6b448d69.png`, ``.

Copy the bundled song and reference images from `example_workflows/assets/` into `ComfyUI/input/`. The example has 20 available clip sections; ensure the song covers every active raw clip.

## NEW - V2V Latent Motion Transfer (with upscale and de-rope)

File: `example_workflows/NEW - V2V Latent Motion Transfer (with upscale and de-rope).json`

Core node metadata: `0.30.0`, `0.33.0`, `0.34.0`.

| External pack | Recorded version/revision |
| --- | --- |
| [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `4ee72c065db22c9d96c2427954dc69e7b908444b` |
| [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | `d7c01b9011f2e8439493f6c02c29995a27df276f` |
| [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `3f20054214fec9f9234fd3841ae6f1e4287948f6` |
| [matlowai/ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) | `c49cde8826da7460361b408bd590dc86d36cc0b2` |

| Loader | Selected file |
| --- | --- |
| UNETLoader | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| MinimaxH3LatentUpscaler3D | `minimax_h3_latent_upscaler_3d_fp16.safetensors` |

Selected input assets: `d47ce26b-c56c-4486-84be-130d728931ca.png`, `90125b58-9ac9-43c9-bd39-461669a40cb8.png`.

Select your own constant-24-fps video with audio. The shipped loader selects 107 frames and skips the first six loaded frames; adjust the selected source interval deliberately. Keep the Pass-1 fractional node and sampler at their documented settings.

## UTILITY - AV Bridge

File: `example_workflows/UTILITY - AV Bridge.json`

Core node metadata: `0.33.0`.

| External pack | Recorded version/revision |
| --- | --- |
| [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `4ee72c065db22c9d96c2427954dc69e7b908444b` |
| [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `073efb07419f56cc714e099a82e49fbc23ad9263` |

| Loader | Selected file |
| --- | --- |
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |

Select two constant-24-fps videos with audio and enough frames for the chosen context. The default canvas is 864×480. The default graph uses source audio sockets directly; for silent files provide matching silence audio before the bridge.

## UTILITY - Custom Keyframes

File: `example_workflows/UTILITY - Custom Keyframes.json`

Core node metadata: `0.30.0`.

| External pack | Recorded version/revision |
| --- | --- |
| No external custom-node pack beyond this repository | ComfyUI core supplies the other nodes |

| Loader | Selected file |
| --- | --- |
| UNETLoader | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |

Selected input assets: `keyframe_1.png`, `keyframe_2.png`, `keyframe_3.png`.

Supply the three still images selected by the example, or replace them. This graph demonstrates soft guides; the masked-keyframe and interior-insert recipes are in [KEYFRAMES_AND_INSERTS.md](KEYFRAMES_AND_INSERTS.md).

## Legacy examples

The three `OLD -` workflows are retained for existing users. Consult their saved loader selections and missing-node list; they are not the recommended Update 9 entry points. Their historical FPS conversion APIs are retained, while current examples enforce 24 fps.

### OLD - Hybrid Extension

Historical metadata only; these graphs were not migrated to the Update 9 workflow controls.

| Pack identifier | Recorded version/revision |
| --- | --- |
| comfy-core | `0.30.0`, `0.31.0` |
| comfyui-kjnodes | `05ba05bf52331622bd3395716c505c29aa1fff76`, `073efb07419f56cc714e099a82e49fbc23ad9263` |
| comfyui-videohelpersuite | `1.7.7`, `4ee72c065db22c9d96c2427954dc69e7b908444b` |
| kijai/ComfyUI-SolAttn_triton | `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b` |
| rgthree-comfy | `6b76ee6f2c5a007710b5a16f97c94330d6ecc871` |

| Loader | Selected file |
| --- | --- |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` |
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |

Replace the saved input assets with your own. Video inputs for H3 should be constant 24 fps.

### OLD - Motion Context - Advanced

Historical metadata only; these graphs were not migrated to the Update 9 workflow controls.

| Pack identifier | Recorded version/revision |
| --- | --- |
| comfy-core | `0.30.0`, `0.31.0` |
| comfyui-kjnodes | `05ba05bf52331622bd3395716c505c29aa1fff76`, `073efb07419f56cc714e099a82e49fbc23ad9263` |
| comfyui-videohelpersuite | `1.7.7` |
| kijai/ComfyUI-SolAttn_triton | `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b` |
| rgthree-comfy | `6b76ee6f2c5a007710b5a16f97c94330d6ecc871` |

| Loader | Selected file |
| --- | --- |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` |
| UNETLoader | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |

Replace the saved input assets with your own. Video inputs for H3 should be constant 24 fps.

### OLD - Motion Context - Simple

Historical metadata only; these graphs were not migrated to the Update 9 workflow controls.

| Pack identifier | Recorded version/revision |
| --- | --- |
| NikoDemon80/ComfyUI-H3-Motion-Context | `15fc6a7bf7b78efb27f33d7eef3818e7ed0e118a` |
| comfy-core | `0.30.0`, `0.31.0` |
| comfyui-kjnodes | `05ba05bf52331622bd3395716c505c29aa1fff76`, `073efb07419f56cc714e099a82e49fbc23ad9263` |
| comfyui-videohelpersuite | `1.7.7` |
| kijai/ComfyUI-SolAttn_triton | `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b` |

| Loader | Selected file |
| --- | --- |
| CLIPLoader | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| LoraLoaderModelOnly | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` |
| UNETLoader | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| VAELoader | `minimax_h3_audio_vae_fp32.safetensors` |
| VAELoader | `minimax_h3_video_vae_int8_convrot.safetensors` |

Replace the saved input assets with your own. Video inputs for H3 should be constant 24 fps.

## Tests

Install `requirements-test.txt`, then run `python tests/run_tests.py` (or the aggregate `pytest -q`). Each module runs in its own pytest process so fixtures execute normally and incompatible ComfyUI mocks cannot leak between modules.
