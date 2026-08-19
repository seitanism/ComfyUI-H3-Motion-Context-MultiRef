"""CPU-only tests for MiniMaxH3CustomKeyframesMasked in nodes.py."""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

_NODES_MODULE_KEY = "update2pkg.nodes_masked_kf_test"


def _install_mocks():
    if "comfy.nested_tensor" in sys.modules:
        return  # comfy mocks already installed

    pkg = types.ModuleType("update2pkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["update2pkg"] = pkg

    # comfy package
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
            samples, size=(height, width), mode="bilinear", align_corners=False,
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


def _install_nodes_mocks():
    _install_mocks()

    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_output_directory = lambda: "/tmp"
        fp.get_save_image_path = lambda prefix, out: (out, prefix, 1, 1, 1)
        sys.modules["folder_paths"] = fp

    if "node_helpers" not in sys.modules:
        nh = types.ModuleType("node_helpers")
        nh.conditioning_set_values = lambda cond, vals, append=False: cond
        sys.modules["node_helpers"] = nh

    if "safetensors" not in sys.modules:
        st = types.ModuleType("safetensors")
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.load_file = None
        st_torch.save_file = None
        sys.modules["safetensors"] = st
        sys.modules["safetensors.torch"] = st_torch

    if "update2pkg.patch_layout" not in sys.modules:
        pl = types.ModuleType("update2pkg.patch_layout")
        pl.MC_KEY = "h3_mc_key"
        pl.MC_AUDIO_KEY = "h3_mc_audio_key"
        sys.modules["update2pkg.patch_layout"] = pl

    # Always patch h3_compat — another test may have installed a version that lacks
    # ensure_motion_context_compat, which nodes.py requires.
    hc = sys.modules.get("update2pkg.h3_compat")
    if hc is None:
        hc = types.ModuleType("update2pkg.h3_compat")
        sys.modules["update2pkg.h3_compat"] = hc
    hc.ensure_existing_video_compat = lambda: True
    hc.ensure_motion_context_compat = lambda: True

    if "update2pkg.h3_auto_crop32" not in sys.modules:
        ha = types.ModuleType("update2pkg.h3_auto_crop32")
        class _Crop: pass
        ha.MiniMaxH3CropTo32 = _Crop
        class _Canvas: pass
        ha.MiniMaxH3StartCanvasSelector = _Canvas
        sys.modules["update2pkg.h3_auto_crop32"] = ha

    if "update2pkg.h3_timing" not in sys.modules:
        ht = types.ModuleType("update2pkg.h3_timing")
        def _crossfade_plan(ctx, req):
            n = max(0, int(ctx))
            eff = min(n, max(0, int(req)))
            return n - eff, eff
        ht.crossfade_plan = _crossfade_plan
        sys.modules["update2pkg.h3_timing"] = ht

    # Load existing_video_extension.py as the real package submodule.
    if "update2pkg.existing_video_extension" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "update2pkg.existing_video_extension",
            ROOT / "existing_video_extension.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)


def _load_nodes():
    _install_nodes_mocks()
    if _NODES_MODULE_KEY in sys.modules:
        return sys.modules[_NODES_MODULE_KEY]
    spec = importlib.util.spec_from_file_location(_NODES_MODULE_KEY, ROOT / "nodes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NODES_MODULE_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────────
# Helpers

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _make_av_latent(video_steps=42, audio_steps=235, h=2, w=4):
    _install_mocks()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    video = torch.zeros((1, 24, video_steps, h, w))
    audio = torch.zeros((1, 32, 2, audio_steps))
    return {"samples": NT((video, audio))}


class _StillVAE:
    """Encodes any input [1,H,W,3] to a single latent step [1,C,1,H//16,W//16]."""
    def encode(self, frames):
        h, w = frames.shape[1], frames.shape[2]
        return torch.ones((1, 24, 1, h // 16, w // 16)) * 0.77


def _make_image(fill=0.5):
    return torch.full((1, 32, 64, 3), fill)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: frame-to-step mapping

def test_phase0_step_mapping():
    """Frames at multiples of 17 map to the still-token (1-frame) step."""
    nodes = _load_nodes()
    # 141-frame latent: 42 steps
    latent_t = 42
    offsets = _step_offsets(latent_t)
    # Phase-0 pixel frames: 0, 17, 34, ...
    for group in range(8):
        pf = group * 17
        # Find expected step: step that starts at pf (the still token)
        expected_step = group * 5
        assert offsets[expected_step] == pf, f"group {group}: offset mismatch"
        assert FRAME_PER_TOKEN[expected_step % 5] == 1, f"group {group}: not a still token"


def test_interior_frame_to_step():
    """Frames inside a 4-frame step map to that step."""
    # step 1 covers frames 1-4; step 2 covers frames 5-8; etc.
    offsets = _step_offsets(10)  # 10 steps
    # Frame 2 should be in step 1 (frames 1-4)
    step_found = None
    for k, off in enumerate(offsets):
        if off <= 2 < off + FRAME_PER_TOKEN[k % 5]:
            step_found = k
            break
    assert step_found == 1, f"Frame 2 should map to step 1, got {step_found}"

    # Frame 5 should be in step 2 (frames 5-8)
    step_found = None
    for k, off in enumerate(offsets):
        if off <= 5 < off + FRAME_PER_TOKEN[k % 5]:
            step_found = k
            break
    assert step_found == 2, f"Frame 5 should map to step 2, got {step_found}"


def test_mask_zeros_exactly_at_pinned_steps():
    nodes = _load_nodes()

    # Pin frames at step boundaries for clean testing.
    # Frame 0 -> step 0 (still token, phase-0)
    # Frame 17 -> step 5 (still token, phase-0)
    latent = _make_av_latent()
    state = '{"count":2,"positions":[1,18]}'  # 1-based: frame 0 and frame 17

    out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
        latent, _StillVAE(), state, "1-based", "disabled",
        keyframe_image_1=_make_image(0.1),
        keyframe_image_2=_make_image(0.2),
    )

    vm, am = out_latent["noise_mask"].unbind()
    # Steps 0 and 5 should be zeroed (preserve).
    assert float(vm[0, 0, 0].max()) == 0.0, "step 0 should be masked"
    assert float(vm[0, 0, 5].max()) == 0.0, "step 5 should be masked"
    # All other video steps should be 1.0 (generate).
    for k in range(42):
        if k in (0, 5):
            continue
        assert float(vm[0, 0, k].min()) == 1.0, f"step {k} should not be masked"


def test_audio_mask_all_ones():
    """Audio must never be masked; it should always be fully generated."""
    nodes = _load_nodes()
    latent = _make_av_latent()
    state = '{"count":1,"positions":[1]}'

    out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
        latent, _StillVAE(), state, "1-based", "disabled",
        keyframe_image_1=_make_image(),
    )

    _, am = out_latent["noise_mask"].unbind()
    assert float(am.min()) == 1.0, "Audio mask must be all-ones (generate everything)"


def test_existing_noise_mask_raises():
    nodes = _load_nodes()
    _install_mocks()
    NT = sys.modules["comfy.nested_tensor"].NestedTensor
    latent = _make_av_latent()
    latent["noise_mask"] = NT((torch.ones(1, 1, 42, 2, 4), torch.ones(1, 1, 2, 235)))
    state = '{"count":1,"positions":[1]}'

    try:
        nodes.MiniMaxH3CustomKeyframesMasked().apply(
            latent, _StillVAE(), state, "1-based", "disabled",
            keyframe_image_1=_make_image(),
        )
        assert False, "Expected ValueError for existing noise_mask"
    except ValueError as e:
        assert "noise_mask" in str(e).lower()


def test_duplicate_step_after_quantization_raises():
    """Two keyframes that map to the same latent step must raise."""
    nodes = _load_nodes()
    latent = _make_av_latent()
    # Frames 1 and 3 both map to step 1 (frames 1-4).
    state = '{"count":2,"positions":[2,4]}'  # 1-based: frames 1 and 3

    try:
        nodes.MiniMaxH3CustomKeyframesMasked().apply(
            latent, _StillVAE(), state, "1-based", "disabled",
            keyframe_image_1=_make_image(0.1),
            keyframe_image_2=_make_image(0.2),
        )
        assert False, "Expected ValueError for duplicate quantized step"
    except ValueError as e:
        msg = str(e).lower()
        assert "latent step" in msg or "step" in msg


def test_all_five_frame_per_token_phases():
    """Verify the frame-to-step mapping covers all 5 phases of FRAME_PER_TOKEN."""
    # Phases: step%5==0 -> 1 frame, step%5 in {1,2,3,4} -> 4 frames
    offsets = _step_offsets(10)
    acc = 0
    for k in range(10):
        expected_off = acc
        assert offsets[k] == expected_off
        acc += FRAME_PER_TOKEN[k % 5]

    # Check coverage: 10 steps cover _pixel_frames(10) frames
    total = _pixel_frames(10)
    # 10 steps: 2 still tokens + 8 four-frame tokens
    # = 2*1 + 8*4 = 34 frames... but cycle is 5: 10 = 2 complete cycles
    # Cycle of 5: 1+4+4+4+4 = 17; 2 cycles = 34
    assert total == 34


def test_0based_indexing():
    """0-based indexing: position 0 = frame 0, position 17 = frame 17 (step 5)."""
    nodes = _load_nodes()
    latent = _make_av_latent()
    state = '{"count":1,"positions":[0]}'  # 0-based: frame 0

    out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
        latent, _StillVAE(), state, "0-based", "disabled",
        keyframe_image_1=_make_image(),
    )

    vm, _ = out_latent["noise_mask"].unbind()
    assert float(vm[0, 0, 0].max()) == 0.0, "Frame 0 (0-based) should pin step 0"


def test_encoded_token_written_to_correct_step():
    """The encoded still token must be written into the correct latent step."""
    nodes = _load_nodes()

    # Use a VAE that fills with a distinctive value.
    class MarkingVAE:
        def encode(self, frames):
            h, w = frames.shape[1], frames.shape[2]
            return torch.full((1, 24, 1, h // 16, w // 16), 0.99)

    latent = _make_av_latent()
    # Frame 17 (1-based pos 18) -> step 5
    state = '{"count":1,"positions":[18]}'

    out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
        latent, MarkingVAE(), state, "1-based", "disabled",
        keyframe_image_1=_make_image(),
    )

    ov, _ = out_latent["samples"].unbind()
    # Step 5 should have value 0.99
    assert torch.allclose(ov[:, :, 5], torch.full_like(ov[:, :, 5], 0.99)), \
        "Encoded token must be written to step 5"
    # Other steps should be 0 (original latent zeros)
    assert float(ov[:, :, 0].max()) == 0.0, "Step 0 should be unchanged (zero)"


class _ListLogHandler(logging.Handler):
    """Minimal handler to capture formatted log records without pytest's caplog.

    The test runner (tests/run_update2_tests.py) calls test functions directly
    with no arguments, so pytest fixtures like caplog are unavailable here.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


def test_1based_position_17_is_interior_static_hold():
    """1-based position 17 -> pixel frame 16 (interior of step 4, frames 13-16);
    not phase-0, so it must trigger the static-hold log with step_span > 1."""
    nodes = _load_nodes()
    latent = _make_av_latent()
    state = '{"count":1,"positions":[17]}'  # 1-based: pixel frame 16

    log = logging.getLogger("h3_motion_context")
    handler = _ListLogHandler()
    log.addHandler(handler)
    prev_level = log.level
    log.setLevel(logging.INFO)
    try:
        out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
            latent, _StillVAE(), state, "1-based", "disabled",
            keyframe_image_1=_make_image(),
        )
    finally:
        log.removeHandler(handler)
        log.setLevel(prev_level)

    # Pixel frame 16 falls in step 4 (frames 13-16, span 4) -> static hold.
    vm, _ = out_latent["noise_mask"].unbind()
    assert float(vm[0, 0, 4].max()) == 0.0, "step 4 should be masked (static hold)"

    hold_msgs = [m for m in handler.records if "static hold" in m]
    assert hold_msgs, "expected a static-hold log line for interior position 17"
    msg = hold_msgs[0]
    assert "pixel frame 16" in msg
    # Nearest phase-0 positions in 1-based indexing: 1 and 18.
    assert "1 and 18" in msg


def test_1based_position_18_is_phase0_no_static_hold():
    """1-based position 18 -> pixel frame 17 (phase-0, step 5); must NOT trigger
    the static-hold log."""
    nodes = _load_nodes()
    latent = _make_av_latent()
    state = '{"count":1,"positions":[18]}'  # 1-based: pixel frame 17

    log = logging.getLogger("h3_motion_context")
    handler = _ListLogHandler()
    log.addHandler(handler)
    prev_level = log.level
    log.setLevel(logging.INFO)
    try:
        out_latent, = nodes.MiniMaxH3CustomKeyframesMasked().apply(
            latent, _StillVAE(), state, "1-based", "disabled",
            keyframe_image_1=_make_image(),
        )
    finally:
        log.removeHandler(handler)
        log.setLevel(prev_level)

    # Pixel frame 17 is the phase-0 still token -> step 5.
    vm, _ = out_latent["noise_mask"].unbind()
    assert float(vm[0, 0, 5].max()) == 0.0, "step 5 should be masked (phase-0)"

    hold_msgs = [m for m in handler.records if "static hold" in m]
    assert not hold_msgs, "phase-0 position 18 must not trigger a static-hold log"
