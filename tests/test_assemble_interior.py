"""CPU-only tests for MiniMaxH3AssembleInterior."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _install_mocks():
    if "comfy.nested_tensor" in sys.modules:
        return  # comfy mocks already installed

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


FPS = 24.0
SR = 32000


def _make_source(n_orig=120, h=32, w=64, source_fps=24.0):
    """Source with n_orig frames at source_fps (distinct pixel values by frame index)."""
    frames = torch.stack([torch.full((h, w, 3), float(i) / n_orig) for i in range(n_orig)])
    audio = {"waveform": torch.rand((1, 2, n_orig * SR // int(source_fps))), "sample_rate": SR}
    return frames, audio, source_fps


def _make_continuation(n_frames=141, h=32, w=64):
    images = torch.rand((n_frames, h, w, 3))
    audio = {
        "waveform": torch.rand((1, 2, round(n_frames / FPS * SR))),
        "sample_rate": SR,
    }
    return images, audio


def test_output_frame_count_unchanged():
    module = _load_module()
    source_frames, source_audio, source_fps = _make_source()
    cont_images, cont_audio = _make_continuation(141)

    node = module.MiniMaxH3AssembleInterior()
    images, audio = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, source_fps,
        insert_frame=51, preserved_frames=22, fps=FPS, crop="disabled",
    )
    assert int(images.shape[0]) == 141, "Output frame count must equal continuation frame count"


def test_audio_samples_match_frame_count():
    module = _load_module()
    source_frames, source_audio, source_fps = _make_source()
    cont_images, cont_audio = _make_continuation(141)

    node = module.MiniMaxH3AssembleInterior()
    images, audio = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, source_fps,
        insert_frame=51, preserved_frames=22, fps=FPS, crop="disabled",
    )
    expected_samples = round(141 / FPS * SR)
    assert int(audio["waveform"].shape[-1]) == expected_samples


def test_spliced_interval_equals_canonical_source_frames():
    """The insert region must contain the canonical last-n source frames (resized)."""
    module = _load_module()

    n_orig = 22  # exactly 22 source frames at 24fps
    h, w = 32, 64
    # Use distinguishable pixel values: frame i fills the image with i/n_orig.
    source_frames = torch.stack([
        torch.full((h, w, 3), float(i) / n_orig) for i in range(n_orig)
    ])
    source_audio = {"waveform": torch.rand((1, 2, round(n_orig / FPS * SR))), "sample_rate": SR}

    cont_images, cont_audio = _make_continuation(141, h=h, w=w)
    n = 22
    insert_frame = 51

    node = module.MiniMaxH3AssembleInterior()
    images, _ = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, 24.0,
        insert_frame=insert_frame, preserved_frames=n, fps=FPS, crop="disabled",
    )

    # The last n canonical source frames are frames 0..21 (all of them at 24fps, 24fps source).
    # After resizing to (h, w), the pixel values should be close to the source's fill values.
    for local_i in range(n):
        expected_fill = float(local_i) / n_orig
        actual = images[insert_frame + local_i]  # [h, w, 3]
        # bilinear resize of a constant-value frame must be that constant
        assert torch.allclose(actual, torch.full_like(actual, expected_fill), atol=1e-4), \
            f"Frame {insert_frame + local_i} (local {local_i}) expected fill {expected_fill:.4f}"


def test_non_insert_region_unchanged():
    """Frames outside the insert interval must equal the continuation images."""
    module = _load_module()
    source_frames, source_audio, _ = _make_source()
    cont_images, cont_audio = _make_continuation(141)

    insert_frame, n = 51, 22
    node = module.MiniMaxH3AssembleInterior()
    images, _ = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, 24.0,
        insert_frame=insert_frame, preserved_frames=n, fps=FPS, crop="disabled",
    )
    assert torch.allclose(images[:insert_frame], cont_images[:insert_frame])
    assert torch.allclose(images[insert_frame + n:], cont_images[insert_frame + n:])


def test_short_source_audio_is_padded_not_raised():
    """Source audio shorter than available frames must be silence-padded, not rejected."""
    module = _load_module()

    n_orig = 120
    h, w = 32, 64
    source_frames = torch.rand((n_orig, h, w, 3))
    # Deliberately give only 1000 samples — far less than 22 frames at 24fps/32000Hz (~29333 samples).
    source_audio = {"waveform": torch.rand((1, 2, 1000)), "sample_rate": SR}

    cont_images, cont_audio = _make_continuation(141, h=h, w=w)

    node = module.MiniMaxH3AssembleInterior()
    # Must not raise; must return correct output shape.
    images, audio = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, 24.0,
        insert_frame=51, preserved_frames=22, fps=FPS, crop="disabled",
    )
    assert int(images.shape[0]) == 141
    assert int(audio["waveform"].shape[-1]) == round(141 / FPS * SR)


def test_splice_uses_same_index_map_as_context_node():
    """The canonical index map used by AssembleInterior must match _cfr_index_map from context."""
    module = _load_module()

    n_orig = 120
    source_frames = torch.arange(n_orig, dtype=torch.float32).reshape(n_orig, 1, 1, 1).expand(n_orig, 4, 4, 3)
    source_audio = {"waveform": torch.rand((1, 2, round(n_orig / FPS * SR))), "sample_rate": SR}
    cont_images, cont_audio = _make_continuation(141, h=4, w=4)

    n = 22
    insert_frame = 17

    # Compute expected canonical tail indices using the module's own _cfr_index_map.
    idx = module._cfr_index_map(n_orig, 24.0, source_frames.device, 24.0)
    tail_idx = idx[-n:]  # last n from the canonical map

    node = module.MiniMaxH3AssembleInterior()
    images, _ = node.assemble(
        cont_images, cont_audio, source_frames, source_audio, 24.0,
        insert_frame=insert_frame, preserved_frames=n, fps=FPS, crop="disabled",
    )

    # Each spliced frame should match the source frame at the corresponding tail index.
    for local_i, src_i in enumerate(tail_idx.tolist()):
        expected_val = float(src_i)
        actual = float(images[insert_frame + local_i, 0, 0, 0])
        assert abs(actual - expected_val) < 0.5, \
            f"Frame {local_i}: expected src index {src_i} (val {expected_val}), got {actual}"
