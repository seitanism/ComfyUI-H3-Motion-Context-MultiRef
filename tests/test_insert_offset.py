"""CPU-only tests for insert_frame offset in MiniMaxH3ExistingVideoMaskedContext."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _install_mocks():
    if "comfy.nested_tensor" in sys.modules:
        return  # comfy mocks already installed (comfy.nested_tensor present)

    pkg = types.ModuleType("update2pkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["update2pkg"] = pkg

    compat = types.ModuleType("update2pkg.h3_compat")
    compat.ensure_existing_video_compat = lambda: True
    sys.modules["update2pkg.h3_compat"] = compat

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
        def process_denoise_mask(self, x): return x
        def scale_latent_inpaint(self, *a, **kw): return None

    model_base_mod.MiniMaxH3 = MiniMaxH3
    comfy.nested_tensor = nested_mod
    comfy.utils = utils_mod
    comfy.model_base = model_base_mod
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested_mod
    sys.modules["comfy.utils"] = utils_mod
    sys.modules["comfy.model_base"] = model_base_mod

    class Functional:
        @staticmethod
        def resample(w, src, dst):
            want = round(w.shape[-1] * dst / src)
            return torch.nn.functional.interpolate(
                w.reshape(-1, 1, w.shape[-1]), size=want, mode="linear", align_corners=False,
            ).reshape(w.shape[0], w.shape[1], want)

    ta = types.ModuleType("torchaudio")
    ta.functional = Functional
    sys.modules["torchaudio"] = ta


def _load_module():
    _install_mocks()
    key = "update2pkg.existing_video_extension"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "existing_video_extension.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_latent(video_steps=42, audio_steps=235, h=2, w=4):
    video = torch.zeros((1, 24, video_steps, h, w))
    audio = torch.zeros((1, 32, 2, audio_steps))
    _install_mocks()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    return {"samples": NT((video, audio))}


class VideoVAE:
    def encode(self, frames):
        n = frames.shape[0]
        t = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
        h, w = frames.shape[1], frames.shape[2]
        return torch.ones((1, 24, t, h // 16, w // 16), dtype=torch.float32) * 0.25


class AudioVAE:
    audio_sample_rate = 32000
    def encode(self, x):
        t = round(x.shape[1] / 32000 * 40)
        return torch.ones((1, 32, 2, t), dtype=torch.float32) * 0.5


def _prepare(module, insert_frame=0, context_length=39, video_steps=42, audio_steps=235,
             audio_feather_ticks=0):
    latent = _make_latent(video_steps=video_steps, audio_steps=audio_steps)
    source_frames = torch.rand((120, 32, 64, 3))
    source_audio = {"waveform": torch.rand((1, 2, 160000)), "sample_rate": 32000}
    node = module.MiniMaxH3ExistingVideoMaskedContext()
    # audio_feather_ticks precedes insert_frame in prepare(); pass both explicitly so
    # insert_frame does not land in the feather slot. Default 0 = hard audio mask.
    return node.prepare(
        latent, VideoVAE(), AudioVAE(),
        source_frames, source_audio, 24.0,
        context_length, "disabled", audio_feather_ticks, insert_frame,
    )


def test_insert_frame_zero_returns_four_outputs():
    module = _load_module()
    result = _prepare(module, insert_frame=0)
    assert len(result) == 4, "Expected 4 outputs: latent, trim_frames, insert_frame, preserved_frames"


def test_insert_frame_zero_identical_prefix_behavior():
    """insert_frame=0 must produce same latent/mask content as baseline prefix node."""
    module = _load_module()
    latent = _make_latent()
    source_frames = torch.rand((120, 32, 64, 3))
    source_audio = {"waveform": torch.rand((1, 2, 160000)), "sample_rate": 32000}
    node = module.MiniMaxH3ExistingVideoMaskedContext()

    out, trim, ins, preserved = node.prepare(
        latent, VideoVAE(), AudioVAE(), source_frames, source_audio,
        24.0, 39, "disabled", 0,
    )

    ov, oa = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    # 39-frame context (smallest AV boundary) -> 12 video steps / 65 audio steps
    assert vm[:, :, :12].max() == 0.0
    assert vm[:, :, 12:].min() == 1.0
    assert am[..., :65].max() == 0.0
    assert am[..., 65:].min() == 1.0
    assert trim == 39
    assert ins == 0
    assert preserved == 39


def test_insert_frame_non_multiple_snaps_down():
    """Non-multiple-of-17 insert_frame snaps down silently (matches context_length behavior)."""
    module = _load_module()
    # insert_frame=10 -> snaps down to 0
    out, trim, ins, preserved = _prepare(module, insert_frame=10)
    assert ins == 0, "insert_frame=10 should snap down to 0"


def test_insert_frame_non_multiple_snaps_down_to_17():
    module = _load_module()
    # insert_frame=20 -> snaps down to 17
    out, trim, ins, preserved = _prepare(module, insert_frame=20)
    assert ins == 17, "insert_frame=20 should snap down to 17"


def test_insert_frame_17():
    """insert_frame=17: s=5, video=[5:17], a_start=28, audio=[28:93]."""
    module = _load_module()
    out, trim, ins, preserved = _prepare(module, insert_frame=17)

    _, _ = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    # s = 17//17 * 5 = 5; video_steps = 12 (39-frame context)
    assert vm[:, :, :5].min() == 1.0, "before insert should be generate"
    assert vm[:, :, 5:17].max() == 0.0, "insert region should be preserve"
    assert vm[:, :, 17:].min() == 1.0, "after insert should be generate"

    # a_start = round(17/24*40) = round(28.33) = 28; audio_steps = 65
    assert am[..., :28].min() == 1.0, "before insert audio should be generate"
    assert am[..., 28:93].max() == 0.0, "insert audio region should be preserve"
    assert am[..., 93:].min() == 1.0, "after insert audio should be generate"

    assert trim == 0, "interior insert should return trim_frames=0"
    assert ins == 17
    assert preserved == 39


def test_insert_frame_51():
    """insert_frame=51: s=15, video=[15:27], a_start=85, audio=[85:150]."""
    module = _load_module()
    out, trim, ins, preserved = _prepare(module, insert_frame=51)

    vm, am = out["noise_mask"].unbind()

    # s = 51//17 * 5 = 15; video_steps = 12 (39-frame context)
    assert vm[:, :, :15].min() == 1.0
    assert vm[:, :, 15:27].max() == 0.0
    assert vm[:, :, 27:].min() == 1.0

    # a_start = round(51/24*40) = 85 (exact); audio_steps = 65
    assert am[..., :85].min() == 1.0
    assert am[..., 85:150].max() == 0.0
    assert am[..., 150:].min() == 1.0

    assert trim == 0
    assert ins == 51
    assert preserved == 39


def test_insert_frame_102():
    """insert_frame=102: s=30, video=[30:42], a_start=170, audio=[170:235].

    Uses a larger target (52 video / 292 audio steps) so a 39-frame preserved
    segment still leaves generate rows after the interior insert.
    """
    module = _load_module()
    out, trim, ins, preserved = _prepare(
        module, insert_frame=102, video_steps=52, audio_steps=292
    )

    vm, am = out["noise_mask"].unbind()

    # s = 102//17 * 5 = 30; video_steps = 12 (39-frame context)
    assert vm[:, :, :30].min() == 1.0
    assert vm[:, :, 30:42].max() == 0.0
    assert vm[:, :, 42:].min() == 1.0

    # a_start = round(102/24*40) = 170 (exact); audio_steps = 65
    assert am[..., :170].min() == 1.0
    assert am[..., 170:235].max() == 0.0
    assert am[..., 235:].min() == 1.0

    assert trim == 0
    assert ins == 102
    assert preserved == 39


def test_insert_frame_zero_trim_equals_n():
    module = _load_module()
    _, trim, _, preserved = _prepare(module, insert_frame=0, context_length=39,
                                      video_steps=42, audio_steps=235)
    # context snaps to 39 (exact H3 run)
    assert trim == preserved == 39


def test_insert_frame_nonzero_trim_is_zero():
    module = _load_module()
    _, trim, _, _ = _prepare(module, insert_frame=17)
    assert trim == 0


def test_full_coverage_insert_zero_produces_all_zero_masks():
    """context_length == target length at insert_frame=0: no generate rows.

    This is the fully-preserved-latent case (e.g. combined with a downstream
    noise_mask edit such as a graduated fade); prepare() must succeed and log
    a warning rather than raise.
    """
    module = _load_module()
    latent = _make_latent(video_steps=42, audio_steps=235)
    source_frames = torch.rand((150, 32, 64, 3))
    # 150 canonical frames at 24 fps -> exactly 200000 samples at 32 kHz; upstream's
    # exact timebase conformance rejects large source-audio/timeline mismatches.
    source_audio = {"waveform": torch.rand((1, 2, 200000)), "sample_rate": 32000}
    node = module.MiniMaxH3ExistingVideoMaskedContext()

    out, trim, ins, preserved = node.prepare(
        latent, VideoVAE(), AudioVAE(), source_frames, source_audio,
        24.0, 141, "disabled", 0,
    )

    vm, am = out["noise_mask"].unbind()

    assert preserved == 141
    assert ins == 0
    assert trim == 141
    assert vm.max() == 0.0
    assert am.max() == 0.0
