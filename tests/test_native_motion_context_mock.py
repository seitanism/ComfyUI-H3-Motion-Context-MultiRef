"""CPU mock for the native #15439 Motion Context payload shape."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def install_module():
    for name in list(sys.modules):
        if name == "nativepkg" or name.startswith("nativepkg."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("nativepkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["nativepkg"] = pkg

    # comfy.utils
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    def common_upscale(samples, width, height, method, crop):
        return torch.nn.functional.interpolate(
            samples, size=(height, width), mode="bilinear", align_corners=False)
    utils.common_upscale = common_upscale
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils

    # folder_paths only needed by unrelated save/load classes at runtime.
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: "/tmp"
    sys.modules["folder_paths"] = folder_paths

    # Minimal conditioning helper matching the stock set-values behavior used here.
    nh = types.ModuleType("node_helpers")
    def conditioning_set_values(conditioning, values, append=False):
        out = []
        for item in conditioning:
            data = dict(item[1])
            for k, v in values.items():
                if append and k in data:
                    data[k] = list(data[k]) + list(v)
                else:
                    data[k] = v
            out.append([item[0], data])
        return out
    nh.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = nh

    # Sibling modules imported by nodes.py but not exercised here.
    compat = types.ModuleType("nativepkg.h3_compat")
    compat.ensure_motion_context_compat = lambda conditioning=None: True
    sys.modules["nativepkg.h3_compat"] = compat

    ext = types.ModuleType("nativepkg.existing_video_extension")
    ext.MiniMaxH3ExistingVideoMaskedContext = type("MiniMaxH3ExistingVideoMaskedContext", (), {})
    ext.MiniMaxH3GeneratedAVMaskedContext = type("MiniMaxH3GeneratedAVMaskedContext", (), {})
    ext.MiniMaxH3StartMaskedContext = type("MiniMaxH3StartMaskedContext", (), {})
    ext.MiniMaxH3AssembleExtension = type("MiniMaxH3AssembleExtension", (), {})
    ext.MiniMaxH3AssembleInterior = type("MiniMaxH3AssembleInterior", (), {})
    ext.MiniMaxH3SetAVNoiseMask = type("MiniMaxH3SetAVNoiseMask", (), {})
    ext.MiniMaxH3ClearAVNoiseMask = type("MiniMaxH3ClearAVNoiseMask", (), {})
    ext._require_h3_mask_support = lambda: True
    sys.modules["nativepkg.existing_video_extension"] = ext

    song = types.ModuleType("nativepkg.h3_song_audio_context")
    song.MiniMaxH3SongMaskedAVContext = type("MiniMaxH3SongMaskedAVContext", (), {})
    sys.modules["nativepkg.h3_song_audio_context"] = song

    crop = types.ModuleType("nativepkg.h3_auto_crop32")
    crop.MiniMaxH3CropTo32 = type("MiniMaxH3CropTo32", (), {})
    crop.MiniMaxH3StartCanvasSelector = type("MiniMaxH3StartCanvasSelector", (), {})
    sys.modules["nativepkg.h3_auto_crop32"] = crop

    timing = types.ModuleType("nativepkg.h3_timing")
    timing.crossfade_plan = lambda context, crossfade: (0, min(context, crossfade))
    sys.modules["nativepkg.h3_timing"] = timing

    spec = importlib.util.spec_from_file_location("nativepkg.nodes", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VideoVAE:
    def encode(self, frames):
        n = int(frames.shape[0])
        t = 1 if n == 1 else (2 if n <= 5 else ((n - 5) // 17) * 5 + 2)
        return torch.ones((1, 24, t, frames.shape[1] // 16, frames.shape[2] // 16))


class AudioVAE:
    audio_sample_rate = 32000
    def encode(self, x):
        # x is [B, samples, channels]
        t = round(int(x.shape[1]) / self.audio_sample_rate * 40)
        return torch.ones((1, 32, 2, t))


def test_native_39_frame_av_guide_preserves_refs():
    module = install_module()
    # 56-frame target -> 17 video latent tokens.
    video = torch.zeros((1, 24, 17, 2, 4))
    audio = torch.zeros((1, 32, 2, round(56 / 24 * 40)))
    latent = {"samples": [video, audio]}
    frames = torch.rand((39, 32, 64, 3))
    sound = {"waveform": torch.rand((1, 2, round(39 / 24 * 32000))), "sample_rate": 32000}
    refs = [
        {"kind": "image", "latent_h": 2, "latent_w": 4},
        {"kind": "audio", "ref_audio_t": 20, "audio_latent": torch.zeros((1,32,2,20))},
    ]
    conditioning = [[None, {"minimax_refs": refs}]]

    node = module.MiniMaxH3MotionContext()
    out, trim = node.apply(
        conditioning, VideoVAE(), latent, frames, 39,
        "video", "head", "disabled", 39, "timeline",
        None, AudioVAE(), sound)

    assert trim == 39
    data = out[0][1]
    assert data["minimax_refs"] is refs
    kfs = data["minimax_keyframes"]
    assert len(kfs) == 1
    assert kfs[0]["resolved_frame_index"] == 0
    assert tuple(kfs[0]["latent"].shape) == (1, 24, 12, 2, 4)
    assert tuple(kfs[0]["audio_latent"].shape) == (1, 32, 2, 65)
    assert "motion_context_index" not in kfs[0]
    assert "motion_context_audio_end_frame" not in kfs[0]
