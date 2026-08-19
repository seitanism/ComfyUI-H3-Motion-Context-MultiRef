"""CPU-only tests for MiniMaxH3SetAVNoiseMask / MiniMaxH3ClearAVNoiseMask."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _install_mocks():
    if "comfy.nested_tensor" in sys.modules:
        return

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
    utils_mod.common_upscale = lambda samples, w, h, m, c: samples

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

    ta = types.ModuleType("torchaudio")
    ta.functional = types.SimpleNamespace(resample=lambda w, s, d: w)
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


def _latent(video_steps=42, audio_steps=235, h=2, w=4, noise_mask=None):
    _install_mocks()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    video = torch.zeros((1, 24, video_steps, h, w))
    audio = torch.zeros((1, 32, 2, audio_steps))
    d = {"samples": NT((video, audio))}
    if noise_mask is not None:
        d["noise_mask"] = noise_mask
    return d


def _frame_mask(frames, h=8, w=16, value=1.0):
    return torch.full((frames, h, w), float(value), dtype=torch.float32)


def test_set_both_masks_round_trips_to_nested_stream_shapes():
    module = _load_module()
    latent = _latent(video_steps=42, audio_steps=235)
    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=_frame_mask(141), audio_mask=_frame_mask(141))
    vm, am = out["noise_mask"].unbind()
    assert vm.shape == (1, 1, 42, 2, 4)
    assert am.shape == (1, 1, 2, 235)


def test_set_video_only_keeps_existing_audio_stream():
    module = _load_module()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    existing_audio = torch.zeros((1, 1, 2, 235))  # hard-preserve audio from insert node
    existing = NT((torch.ones((1, 1, 42, 2, 4)), existing_audio))
    latent = _latent(noise_mask=existing)

    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=_frame_mask(141, value=1.0), audio_mask=None)
    vm, am = out["noise_mask"].unbind()
    # Audio stream is the untouched existing one.
    assert am is existing_audio
    assert vm.shape == (1, 1, 42, 2, 4)


def test_set_audio_only_keeps_existing_video_stream():
    module = _load_module()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    existing_video = torch.ones((1, 1, 42, 2, 4))
    existing = NT((existing_video, torch.zeros((1, 1, 2, 235))))
    latent = _latent(noise_mask=existing)

    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=None, audio_mask=_frame_mask(141))
    vm, am = out["noise_mask"].unbind()
    assert vm is existing_video
    assert am.shape == (1, 1, 2, 235)


def test_set_both_none_raises_pointing_at_clear_node():
    module = _load_module()
    latent = _latent()
    node = module.MiniMaxH3SetAVNoiseMask()
    try:
        node.set_mask(latent, video_mask=None, audio_mask=None)
    except ValueError as exc:
        assert "MiniMaxH3ClearAVNoiseMask" in str(exc)
    else:
        raise AssertionError("both-None must raise")


def test_set_none_without_existing_stream_defaults_to_all_generate():
    """video-only Set on a mask-less latent leaves audio all-generate (all-ones).

    H3 ignores an all-ones stream (min == 1), so this is the same as 'no audio
    preservation' without erroring -- the Clear -> Set(video only) workflow.
    """
    module = _load_module()
    latent = _latent()  # no noise_mask present
    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=_frame_mask(141), audio_mask=None)
    vm, am = out["noise_mask"].unbind()
    assert am.shape == (1, 1, 2, 235)
    assert am.min() == 1.0  # all-generate: H3 treats this as absent
    assert vm.shape == (1, 1, 42, 2, 4)


def test_clear_then_set_video_only_round_trips_without_error():
    """The reported workflow: Clear AV Noise, then Set with video only."""
    module = _load_module()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    existing = NT((torch.zeros((1, 1, 42, 2, 4)), torch.zeros((1, 1, 2, 235))))
    latent = _latent(noise_mask=existing)

    cleared = module.MiniMaxH3ClearAVNoiseMask().clear_mask(latent)[0]
    assert "noise_mask" not in cleared

    (out,) = module.MiniMaxH3SetAVNoiseMask().set_mask(
        cleared, video_mask=_frame_mask(141), audio_mask=None
    )
    vm, am = out["noise_mask"].unbind()
    assert vm.shape == (1, 1, 42, 2, 4)
    assert am.min() == 1.0  # audio fully regenerates, as intended after a Clear


def test_clear_then_set_audio_only_round_trips_without_error():
    """Symmetric to the video-only case: Clear, then Set with audio only."""
    module = _load_module()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    existing = NT((torch.zeros((1, 1, 42, 2, 4)), torch.zeros((1, 1, 2, 235))))
    latent = _latent(noise_mask=existing)

    cleared = module.MiniMaxH3ClearAVNoiseMask().clear_mask(latent)[0]
    assert "noise_mask" not in cleared

    (out,) = module.MiniMaxH3SetAVNoiseMask().set_mask(
        cleared, video_mask=None, audio_mask=_frame_mask(141)
    )
    vm, am = out["noise_mask"].unbind()
    assert am.shape == (1, 1, 2, 235)
    assert vm.min() == 1.0  # video fully regenerates, as intended after a Clear


def test_set_preserves_samples_and_uses_a_copy():
    module = _load_module()
    latent = _latent()
    samples_before = latent["samples"]
    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=_frame_mask(141), audio_mask=_frame_mask(141))
    assert out["samples"] is samples_before  # untouched
    assert "noise_mask" not in latent         # original dict not mutated


def test_set_steep_ramp_keeps_distinct_per_step_values():
    module = _load_module()
    latent = _latent(video_steps=42, audio_steps=235)
    # Steep frame-space ramp 0..1 across 141 frames (uniform per frame spatially).
    ramp = torch.linspace(0.0, 1.0, 141).view(141, 1, 1).repeat(1, 8, 16)
    node = module.MiniMaxH3SetAVNoiseMask()
    (out,) = node.set_mask(latent, video_mask=ramp, audio_mask=ramp)
    vm, am = out["noise_mask"].unbind()
    v_series = vm[0, 0, :, 0, 0]
    a_series = am[0, 0, 0, :]
    assert v_series.min() < 0.05 and v_series.max() > 0.95
    assert torch.unique(v_series).numel() > 20   # no collapse to a hard cut
    assert torch.unique(a_series).numel() > 100
    # audio channels are identical broadcasts
    assert torch.equal(am[0, 0, 0], am[0, 0, 1])


def test_clear_removes_noise_mask_and_keeps_samples():
    module = _load_module()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    existing = NT((torch.ones((1, 1, 42, 2, 4)), torch.zeros((1, 1, 2, 235))))
    latent = _latent(noise_mask=existing)
    samples_before = latent["samples"]

    node = module.MiniMaxH3ClearAVNoiseMask()
    (out,) = node.clear_mask(latent)
    assert "noise_mask" not in out
    assert out["samples"] is samples_before
    assert "noise_mask" in latent  # original dict not mutated
