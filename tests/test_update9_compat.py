"""Lazy/native retirement tests for PRs #15972 and #15988."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec);sys.modules[name] = m;spec.loader.exec_module(m)
    return m


def audio_mock(monkeypatch):
    class H3Audio:
        pass
    mod = types.ModuleType("comfy.ldm.minimax.audio_vae")
    mod.MiniMaxH3AudioVAE = H3Audio
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    class VAE:
        def __init__(self, native=False):
            self.first_stage_model = H3Audio()
            self.crop_input = not native
        def encode(self, x):
            if self.crop_input:
                extra = x.shape[1] % 800
                x = x[:, extra//2:extra//2 + x.shape[1]//800*800]
            self.origin = x[0, 0, 0].item()
            return torch.zeros(1, 32, 2, (x.shape[1] + 799)//800)
    return VAE


def test_audio_crop_fix_is_lazy_instance_scoped_and_native_safe(monkeypatch):
    VAE = audio_mock(monkeypatch)
    m = module_at("u9_audio_compat", ROOT / "h3_audio_vae_compat.py")
    pcm = torch.arange(437333, dtype=torch.float32).reshape(1, -1, 1)
    old, other, native = VAE(), VAE(), VAE(True)
    assert old.encode(pcm).shape[-1] == 546 and old.origin == 266
    assert old.crop_input and other.crop_input
    assert m.ensure_h3_audio_vae_crop(old)["patched"]
    assert old.encode(pcm).shape[-1] == 547 and old.origin == 0
    assert other.crop_input
    assert m.ensure_h3_audio_vae_crop(old)["patched"]
    assert not m.ensure_h3_audio_vae_crop(native)["patched"]
    assert native.encode(pcm).shape[-1] == 547 and native.origin == 0
    unrelated = types.SimpleNamespace(first_stage_model=object(), crop_input=True)
    assert not m.ensure_h3_audio_vae_crop(unrelated)["is_h3_audio"]
    assert unrelated.crop_input


def velocity_mock(monkeypatch, native):
    comfy = types.ModuleType("comfy");ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax");h3m = types.ModuleType("comfy.ldm.minimax.model")
    patcher = types.ModuleType("comfy.patcher_extension")
    patcher.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion")
    patcher.get_all_wrappers = lambda *args: []
    patcher.WrapperExecutor = types.SimpleNamespace(new_class_executor=lambda fn, *args: types.SimpleNamespace(execute=fn))
    comfy.model_prefetch = types.SimpleNamespace(malloc_graph_enabled=lambda device: False)
    comfy.patcher_extension = patcher;comfy.ldm = ldm;ldm.minimax = minimax;minimax.model = h3m
    for m in (comfy, ldm, minimax, h3m, patcher):monkeypatch.setitem(sys.modules, m.__name__, m)
    fixture = module_at("u9_forward_fixture", ROOT / "tests/fixtures/h3_forward_15988.py")
    h3m.MiniMaxH3Model = fixture.FixedH3 if native else fixture.LegacyH3
    h3m.time_shift_sigma = fixture.time_shift_sigma
    return h3m


def test_velocity_patch_matches_upstream_and_retires_natively(monkeypatch):
    m = module_at("u9_velocity", ROOT / "h3_mask_velocity.py")
    for native in (False, True):
        h3m = velocity_mock(monkeypatch, native)
        before = h3m.MiniMaxH3Model.forward
        assert m.capability_status()["velocity_ready"] is native
        result = m.ensure_h3_mask_velocity()
        assert result["velocity_ready"]
        assert result["patch_active"] is (not native)
        installed = h3m.MiniMaxH3Model.forward
        m.ensure_h3_mask_velocity()
        assert h3m.MiniMaxH3Model.forward is installed
        if native:assert installed is before


def test_native_probe_covers_audio_carry_order(monkeypatch):
    h3m = velocity_mock(monkeypatch, True)
    m = module_at("u9_velocity_order", ROOT / "h3_mask_velocity.py")
    # Multiplication outside native forward doubles the mask and scales the
    # audio carry incorrectly. The behavioral probe must reject it.
    native = h3m.MiniMaxH3Model.forward
    def wrong(self, *args, **kwargs):
        out = native(self, *args, **kwargs)
        out[1] *= kwargs["audio_denoise_mask"]
        return out
    h3m.MiniMaxH3Model.forward = wrong
    assert not m.capability_status()["velocity_ready"]
    with pytest.raises(RuntimeError, match="closure"):
        m.ensure_h3_mask_velocity()
    assert h3m.MiniMaxH3Model.forward is wrong


def test_velocity_patch_preserves_malloc_output_copy(monkeypatch):
    h3m = velocity_mock(monkeypatch, False)
    calls = []
    prefetch = sys.modules["comfy"].model_prefetch
    prefetch.malloc_graph_enabled = lambda device: True
    prefetch.malloc_graph_begin = lambda *args: calls.append("begin")
    prefetch.malloc_graph_end = lambda: calls.append("end")
    m = module_at("u9_velocity_malloc", ROOT / "h3_mask_velocity.py")
    assert m.ensure_h3_mask_velocity()["velocity_ready"]
    assert calls and calls == ["begin", "end"] * (len(calls) // 2)
