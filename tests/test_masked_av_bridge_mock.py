"""CPU-only regression for two-ended PR #15375-style H3 AV bridge masks."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]

pkg = types.ModuleType("bridgepkg")
pkg.__path__ = [str(ROOT)]
sys.modules["bridgepkg"] = pkg

# Minimal comfy package used by the bridge + shared existing-video helpers.
comfy = types.ModuleType("comfy")
nested_mod = types.ModuleType("comfy.nested_tensor")
utils_mod = types.ModuleType("comfy.utils")
model_base_mod = types.ModuleType("comfy.model_base")


class NestedTensor:
    def __init__(self, xs):
        self.xs = list(xs)

    def unbind(self):
        return tuple(self.xs)

    @property
    def is_nested(self):
        return True


nested_mod.NestedTensor = NestedTensor


def common_upscale(samples, width, height, method, crop):
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode="bilinear", align_corners=False
    )


utils_mod.common_upscale = common_upscale


class MiniMaxH3:
    def process_denoise_mask(self, x):
        return x

    def scale_latent_inpaint(self, *args, **kwargs):
        return None


model_base_mod.MiniMaxH3 = MiniMaxH3
comfy.nested_tensor = nested_mod
comfy.utils = utils_mod
comfy.model_base = model_base_mod
sys.modules["comfy"] = comfy
sys.modules["comfy.nested_tensor"] = nested_mod
sys.modules["comfy.utils"] = utils_mod
sys.modules["comfy.model_base"] = model_base_mod

# Existing-video helper imports this compatibility entry point. In the unit test,
# the fake MiniMaxH3 class above already exposes the two required methods.
compat = types.ModuleType("bridgepkg.h3_compat")
compat.ensure_existing_video_compat = lambda: True
sys.modules["bridgepkg.h3_compat"] = compat

# Length-correct resampling stub.
class Functional:
    @staticmethod
    def resample(w, src, dst):
        want = round(w.shape[-1] * dst / src)
        return torch.nn.functional.interpolate(
            w.reshape(-1, 1, w.shape[-1]),
            size=want,
            mode="linear",
            align_corners=False,
        ).reshape(w.shape[0], w.shape[1], want)


ta = types.ModuleType("torchaudio")
ta.functional = Functional
sys.modules["torchaudio"] = ta

# Load shared helper module first, then the bridge module.
for name in ("h3_timing", "existing_video_extension", "h3_masked_bridge"):
    spec = importlib.util.spec_from_file_location(
        f"bridgepkg.{name}", ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

bridge = sys.modules["bridgepkg.h3_masked_bridge"]


class VideoVAE:
    def encode(self, frames):
        n = int(frames.shape[0])
        t = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
        h, w = int(frames.shape[1]), int(frames.shape[2])
        # Encode source identity into the mock latent so prefix/suffix placement
        # can be asserted independently.
        value = float(frames.mean())
        return torch.full((1, 24, t, h // 16, w // 16), value)


class AudioVAE:
    audio_sample_rate = 32000
    first_stage_model = SimpleNamespace(samples_per_latent=800)

    def __init__(self):
        self.calls = []

    def encode(self, x):
        # Model the current generic Comfy VAE wrapper: non-grid input would be
        # center-cropped before H3 sees it. Exact-grid callers must make offset 0.
        length = int(x.shape[1])
        cropped = (length // 800) * 800
        offset = (length % 800) // 2
        self.calls.append({
            "length": length,
            "crop_offset": offset,
            "first": float(x[0, 0, 0]),
            "last": float(x[0, length - 1, 0]),
        })
        t = cropped // 800
        value = float(x.mean())
        return torch.full((1, 32, 2, t), value)


def test_192_frame_bridge_preserves_39_at_both_ends():
    # 192 H3 frames -> 57 video latent steps and exactly 320 audio latent steps.
    video = torch.zeros((1, 24, 57, 2, 4))
    audio = torch.zeros((1, 32, 2, 320))
    latent = {"samples": NestedTensor((video, audio))}

    start_frames = torch.full((100, 32, 64, 3), 0.2)
    end_frames = torch.full((100, 32, 64, 3), 0.8)
    start_audio = {
        "waveform": torch.full((1, 2, round(100 / 24 * 32000)), 0.3),
        "sample_rate": 32000,
    }
    end_audio = {
        "waveform": torch.full((1, 2, round(100 / 24 * 32000)), 0.7),
        "sample_rate": 32000,
    }

    node = bridge.MiniMaxH3MaskedAVBridge()
    out, middle, preserved = node.prepare(
        latent,
        VideoVAE(),
        AudioVAE(),
        start_frames,
        start_audio,
        end_frames,
        end_audio,
        24.0,
        24.0,
        39,
        "disabled",
    )

    assert middle == 114
    assert preserved == 39

    ov, oa = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    assert ov.shape == (1, 24, 57, 2, 4)
    assert oa.shape == (1, 32, 2, 320)

    # 39 frames encode to 12 video latent steps; the exact middle stays as the
    # target latent supplied by MiniMax H3 Image-to-Video (zeros in this mock).
    assert torch.allclose(ov[:, :, :12], torch.full_like(ov[:, :, :12], 0.2))
    assert torch.count_nonzero(ov[:, :, 12:-12]) == 0
    assert torch.allclose(ov[:, :, -12:], torch.full_like(ov[:, :, -12:], 0.8))

    # 39 / 24 * 40 = 65 exact audio steps per side.
    assert torch.allclose(oa[..., :65], torch.full_like(oa[..., :65], 0.3))
    assert torch.count_nonzero(oa[..., 65:-65]) == 0
    assert torch.allclose(oa[..., -65:], torch.full_like(oa[..., -65:], 0.7))

    assert vm[:, :, :12].max() == 0
    assert vm[:, :, 12:-12].min() == 1
    assert vm[:, :, -12:].max() == 0
    assert am[..., :65].max() == 0
    assert am[..., 65:-65].min() == 1
    assert am[..., -65:].max() == 0


def test_rejects_non_h3_preserve_length():
    video = torch.zeros((1, 24, 57, 2, 4))
    audio = torch.zeros((1, 32, 2, 320))
    latent = {"samples": NestedTensor((video, audio))}
    frames = torch.zeros((100, 32, 64, 3))
    a = {"waveform": torch.zeros((1, 2, round(100 / 24 * 32000))), "sample_rate": 32000}

    node = bridge.MiniMaxH3MaskedAVBridge()
    try:
        node.prepare(latent, VideoVAE(), AudioVAE(), frames, a, frames, a, 24.0, 24.0, 40, "disabled")
    except ValueError as exc:
        assert "shared AV boundary" in str(exc)
    else:
        raise AssertionError("40-frame preserve length should have been rejected")


def test_nonshared_bridge_context_is_rejected():
    for n in (5, 22, 56, 73):
        try:
            bridge._validate_preserve_frames(n, 500, 500, 1000)
        except ValueError as exc:
            assert "39 + 51*k" in str(exc)
        else:
            raise AssertionError(f"{n} is not an exact shared AV context")
    for n in (39, 90, 141, 192):
        assert bridge._validate_preserve_frames(n, 500, 500, 1000) == n
