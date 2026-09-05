"""CPU mock for the native #15439 Motion Context payload shape."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = lambda xs: xs
    comfy.nested_tensor = nested
    sys.modules["comfy.nested_tensor"] = nested
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
    ext.MiniMaxH3SourceAudioRegenLength = type("MiniMaxH3SourceAudioRegenLength", (), {})
    ext.MiniMaxH3SourceAudioRegenMask = type("MiniMaxH3SourceAudioRegenMask", (), {})
    ext.MiniMaxH3SourceAudioPolicy = type("MiniMaxH3SourceAudioPolicy", (), {})
    ext.MiniMaxH3AssembleInterior = type("MiniMaxH3AssembleInterior", (), {})
    ext.MiniMaxH3SetAVNoiseMask = type("MiniMaxH3SetAVNoiseMask", (), {})
    ext.MiniMaxH3ClearAVNoiseMask = type("MiniMaxH3ClearAVNoiseMask", (), {})
    ext.MiniMaxH3FanRecoveredContext = type("MiniMaxH3FanRecoveredContext", (), {})
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


class CenterCropContextAudioVAE:
    audio_sample_rate = 32000
    first_stage_model = SimpleNamespace(samples_per_latent=800)

    def __init__(self):
        self.last_input = None
        self.last_crop_offset = None
        self.first_sample = None
        self.last_sample = None

    def encode(self, x):
        length = int(x.shape[1])
        self.last_input = length
        self.last_crop_offset = (length % 800) // 2
        self.first_sample = float(x[0, 0, 0])
        self.last_sample = float(x[0, length - 1, 0])
        cropped = (length // 800) * 800
        return torch.ones((1, 32, 2, cropped // 800))


def test_offgrid_motion_audio_context_is_exact_grid_and_seam_aligned():
    module = install_module()
    # 73-frame target; preserve a 56-frame native video run. 56 is off the
    # shared audio boundary and would previously suffer a 133-sample center crop.
    video = torch.zeros((1, 24, 22, 2, 4))
    audio = torch.zeros((1, 32, 2, round(73 / 24 * 40)))
    latent = {"samples": [video, audio]}
    frames = torch.rand((56, 32, 64, 3))
    picture = round(56 / 24 * 32000)
    wave = torch.arange(picture, dtype=torch.float32).reshape(1, 1, -1).repeat(1, 2, 1)
    sound = {"waveform": wave, "sample_rate": 32000}
    av = CenterCropContextAudioVAE()
    conditioning = [[None, {}]]

    out, trim = module.MiniMaxH3MotionContext().apply(
        conditioning, VideoVAE(), latent, frames, 56,
        "video", "head", "disabled", 56, "timeline",
        None, av, sound,
    )

    assert trim == 56
    assert av.last_input == 93 * 800 == 74400
    assert av.last_crop_offset == 0
    # End alignment keeps the continuation seam exact; only the older outer
    # edge moves by 267 samples because 56f is not a shared AV boundary.
    assert av.first_sample == 267.0
    assert av.last_sample == float(picture - 1)
    kf = out[0][1]["minimax_keyframes"][0]
    assert kf["audio_latent"].shape[-1] == 93


def test_motion_context_trim_time_conforms_round_down_and_round_up_audio():
    module = install_module()
    module.torchaudio = None  # deterministic dependency-free interpolation path
    node = module.MiniMaxH3MotionContextTrim()

    # Bundled legacy workflows use 362 frames. H3 target audio rounds
    # 362/24*40 = 603.333... to 603 cells, so decoded audio is 267 PCM
    # samples short at 32 kHz. The old code left that undershoot untouched.
    images = torch.zeros((362, 4, 4, 3), dtype=torch.float32)
    short_audio = {
        "waveform": torch.zeros((1, 2, 603 * 800), dtype=torch.float32),
        "sample_rate": 32000,
    }
    _images, fixed_short, _xfade, _n = node.trim(
        images, 0, short_audio, fps=24.0, match_tail=True, video_crossfade_frames=39
    )
    assert fixed_short["waveform"].shape[-1] == round(362 / 24 * 32000)

    # 124 frames is the opposite direction: 206.666... -> 207 cells,
    # so decoded audio is 267 samples long and must also conform exactly.
    images2 = torch.zeros((124, 4, 4, 3), dtype=torch.float32)
    long_audio = {
        "waveform": torch.zeros((1, 2, 207 * 800), dtype=torch.float32),
        "sample_rate": 32000,
    }
    _images2, fixed_long, _xfade2, _n2 = node.trim(
        images2, 0, long_audio, fps=24.0, match_tail=True, video_crossfade_frames=39
    )
    assert fixed_long["waveform"].shape[-1] == round(124 / 24 * 32000)


def test_native_motion_context_can_be_placed_at_an_interior_target_offset():
    module = install_module()
    video = torch.zeros((1, 24, 117, 2, 4))  # 396 pixel frames
    audio = torch.zeros((1, 32, 2, round(396 / 24 * 40)))
    latent = {"samples": [video, audio]}
    frames = torch.rand((39, 32, 64, 3))

    out, trim = module.MiniMaxH3MotionContext().apply(
        [[None, {}]], VideoVAE(), latent, frames, 39,
        "video", "head", "disabled", 0, "timeline",
        target_start=116,
    )

    assert trim == 0
    kfs = out[0][1]["minimax_keyframes"]
    assert len(kfs) == 1
    assert kfs[0]["resolved_frame_index"] == 116
    assert tuple(kfs[0]["latent"].shape) == (1, 24, 12, 2, 4)


def test_motion_context_offset_rejects_span_that_consumes_target_tail():
    module = install_module()
    video = torch.zeros((1, 24, 17, 2, 4))
    audio = torch.zeros((1, 32, 2, round(56 / 24 * 40)))
    latent = {"samples": [video, audio]}
    frames = torch.rand((39, 32, 64, 3))
    try:
        module.MiniMaxH3MotionContext().apply(
            [[None, {}]], VideoVAE(), latent, frames, 39,
            "video", "head", "disabled", 0, "timeline",
            target_start=17,
        )
    except ValueError as exc:
        assert "does not leave room" in str(exc)
    else:
        raise AssertionError("expected target_start span validation to fail")


def test_offset_timeline_audio_moves_with_visual_guide_and_keeps_exact_grid():
    module = install_module()
    video = torch.zeros((1, 24, 117, 2, 4))
    audio = torch.zeros((1, 32, 2, round(396 / 24 * 40)))
    latent = {"samples": [video, audio]}
    frames = torch.rand((39, 32, 64, 3))
    sound = {
        "waveform": torch.rand((1, 2, round(39 / 24 * 32000))),
        "sample_rate": 32000,
    }
    out, trim = module.MiniMaxH3MotionContext().apply(
        [[None, {}]], VideoVAE(), latent, frames, 39,
        "video", "head", "disabled", 39, "timeline",
        None, AudioVAE(), sound, target_start=116,
    )
    assert trim == 0
    kfs = out[0][1]["minimax_keyframes"]
    assert len(kfs) == 1
    assert kfs[0]["resolved_frame_index"] == 116
    assert kfs[0]["audio_latent"].shape[-1] == 65


def test_target_start_is_appended_for_backward_widget_serialization():
    module = install_module()
    required = list(module.MiniMaxH3MotionContext.INPUT_TYPES()["required"].keys())
    assert required[-1] == "target_start"

    import inspect
    params = list(inspect.signature(module.MiniMaxH3MotionContext.apply).parameters)
    assert params[-1] == "target_start"
