"""Update 9 CPU regressions for bugs not covered by snapshot/string tests."""
import importlib.util
from datetime import datetime
from pathlib import Path
import sys
import types
import weakref

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    pkg = types.ModuleType("u9pkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["u9pkg"] = pkg
    spec = importlib.util.spec_from_file_location("u9pkg." + name, ROOT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_first_probe_owns_only_one_frame_storage():
    m = load("h3_streaming_vhs")
    refs = []
    def decode(*args):
        if refs:
            assert refs[0]() is None, "first full clip must be released before decoding clip 2"
        frames = torch.zeros(39, 32, 64, 3)
        refs.append(weakref.ref(frames))
        return frames
    m._decode_h3_video_cpu = decode
    m._release_decode_memory = lambda: None
    seq = m._OneShotFrameSequence(78, lambda: m._generated_frame_generator(
        None, [None, None], [39, 39], [0, 0], 0, "test"))
    vhs_first_image = seq[0]
    assert vhs_first_image.untyped_storage().nbytes() == vhs_first_image.numel() * 4
    count = 0
    iterator = iter(seq)
    while True:
        try:
            # VHS converts each tensor to encoder bytes before requesting another.
            next(iterator).numpy().tobytes()
            count += 1
        except StopIteration:
            break
    assert count == 78
    assert vhs_first_image.shape == (32, 64, 3)


def test_spatial_mask_follows_both_center_crop_axes():
    m = load("h3_v2v_fractional")
    for h, w in ((64, 128), (128, 64)):
        mask = torch.zeros(5, h, w)
        if w > h:
            mask[..., :32] = 1
        else:
            mask[:, :32, :] = 1
        cropped = m._mask_to_video_latent(mask, 2, 4, 4, "max", "center")
        stretched = m._mask_to_video_latent(mask, 2, 4, 4, "max", "disabled")
        assert not cropped.any()
        assert stretched.any()


def test_date_output_folders_work_without_browser_hooks():
    m = load("h3_streaming_vhs")
    date = datetime(2026, 9, 5, 16, 7, 9)
    assert m._expand_output_prefix("%date:yyyy-MM-dd%/MiniMax_H3_", date) == "2026-09-05/MiniMax_H3_"
    assert m._expand_output_prefix("video/%date:HH-mm-ss%/clip", date) == "video/16-07-09/clip"
    assert m._expand_output_prefix("video/clip", date) == "video/clip"
    for path in ("../clip", "/tmp/clip", "C:\\clip", "%date:QQ%/clip", "x\x00y"):
        with pytest.raises(ValueError):
            m._expand_output_prefix(path, date)


def test_24fps_metadata_validator_rejects_loader_retiming():
    m = load("h3_workflow_utilities")
    node = m.MiniMaxH3Validate24FPSVideo()
    frames = object()
    assert node.validate(frames, {"source_fps": 24, "loaded_fps": 24}) == (frames,)
    for info in ({"source_fps": 30, "loaded_fps": 24}, {"source_fps": 24, "loaded_fps": 12}, {}):
        with pytest.raises(ValueError, match="24 fps"):
            node.validate(frames, info)


def test_bridge_timing_accepts_video_runs_but_only_shared_av_contexts():
    m = load("h3_workflow_utilities")
    assert m._bridge_timing(192, 39) == (192, 39)
    assert m._bridge_timing(107, 39) == (107, 39)
    assert m._bridge_timing(396, 192) == (396, 192)
    for target, context in ((193, 39), (192, 56), (175, 90), (192, 0)):
        with pytest.raises(ValueError):
            m._bridge_timing(target, context)


def test_short_timeline_audio_moves_to_seam_without_changing_complete_guides():
    spec = importlib.util.spec_from_file_location("u9_native_mock", ROOT / "tests/test_native_motion_context_mock.py")
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    m = t.install_module()
    latent = {"samples": [torch.zeros(1, 24, 17, 2, 4), torch.zeros(1, 32, 2, 93)]}
    for samples, expected_frame in ((32000, 15), (52000, 0)):
        sound = {"waveform": torch.ones(1, 2, samples), "sample_rate": 32000}
        out, trim = m.MiniMaxH3MotionContext().apply(
            [[None, {}]], t.VideoVAE(), latent, torch.ones(39, 32, 64, 3), 39,
            "video", "head", "disabled", 39, "timeline", None, t.AudioVAE(), sound)
        guide = next(k for k in out[0][1]["minimax_keyframes"] if "audio_latent" in k)
        assert guide["resolved_frame_index"] == expected_frame
        assert trim == 39


def test_bridge_audio_conforms_nonshared_targets_to_absolute_picture_timeline():
    import importlib
    spec = importlib.util.spec_from_file_location("u9_interior_mock", ROOT / "tests/test_assemble_interior.py")
    t = importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    # Establish the existing minimal ComfyUI mocks in this test process.
    sys.modules.pop("comfy.nested_tensor", None)
    t._install_mocks()
    ext = load("existing_video_extension")
    m = load("h3_workflow_utilities")
    ext.torchaudio = None
    sr = 44100
    for target, preserve in ((192, 39), (107, 39), (124, 39), (396, 192)):
        n1, n2 = 205, 207
        ticks, ctx = round(target/24*40), preserve*5//3
        start = {"sample_rate": sr, "waveform": torch.ones(1, 2, round(n1/24*sr))}
        end = {"sample_rate": sr, "waveform": torch.full((1, 2, round(n2/24*sr)), 3.)}
        raw = torch.full((1, 2, round(ticks/40*sr)), 9.)
        raw[..., round(ctx/40*sr):round((ticks-ctx)/40*sr)] = 2.
        result, = m.MiniMaxH3AssembleBridgeAudio().assemble(
            {"sample_rate": sr, "waveform": raw}, start, end,
            torch.empty(n1, 1, 1, 3), torch.empty(n2, 1, 1, 3), target, preserve)
        a, b, total = [round(f/24*sr) for f in (n1, n1+target-2*preserve, n1+target-2*preserve+n2)]
        assert result["waveform"].shape[-1] == total
        assert torch.all(result["waveform"][..., :a] == 1)
        torch.testing.assert_close(result["waveform"][..., a:b], torch.full((1, 2, b-a), 2.))
        # Relative-to-absolute source alignment can pad at most one final sample.
        assert torch.all(result["waveform"][..., b:-1] == 3)
