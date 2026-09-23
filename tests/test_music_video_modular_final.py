from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "h3_streaming_vhs.py"
WF = ROOT / "example_workflows" / "NEW - Music Video.json"
JS_INPUTS = ROOT / "js" / "h3_stream_extension_inputs.js"
JS_CONTROLLER = ROOT / "js" / "h3_music_video_controller.js"


def _load_stream_module():
    spec = spec_from_file_location("_h3_music_modular_stream_test_module", STREAM)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input(node, name):
    return next(i for i in node.get("inputs", []) if i["name"] == name)


def test_music_final_stream_is_dynamic_and_supports_standalone_trailing_empty_inputs():
    module = _load_stream_module()
    cls = module.MiniMaxH3StreamLiveMusicVideoToVHS
    schema = cls.INPUT_TYPES()
    assert cls.MAX_CLIPS == 64
    assert schema["required"]["input_count"][1]["default"] == 20
    assert schema["required"]["input_count"][1]["max"] == 64
    assert "active_clips" in schema["optional"]
    assert "preview_gate" in schema["optional"]
    for i in (1, 20, 32, 64):
        assert schema["optional"][f"clip_{i}"][0] == "LATENT"
        assert schema["optional"][f"clip_{i}"][1]["lazy"] is True

    node = cls()
    common = dict(
        video_vae=object(), master_audio=object(), input_count=6,
        context_frames=39, video_overlap_frames=39,
        filename_prefix="x", pix_fmt="yuv420p", crf=19,
        save_metadata=False, trim_to_audio=False, save_output=True,
        active_clips=None,
    )
    # Only the connected prefix is requested. Trailing sockets can stay empty.
    assert node.check_lazy_status(**common, clip_1=None, clip_2=None) == ["clip_1", "clip_2"]
    assert node._validate_connected_prefix(
        6, None, {"clip_1": object(), "clip_2": object()}
    ) == [1, 2]


def test_music_final_stream_rejects_middle_holes_to_preserve_master_song_timing():
    module = _load_stream_module()
    node = module.MiniMaxH3StreamLiveMusicVideoToVHS()
    try:
        node._validate_connected_prefix(
            6, None,
            {"clip_1": object(), "clip_2": object(), "clip_4": object()},
        )
    except ValueError as exc:
        text = str(exc)
        assert "contiguous" in text
        assert "missing clip_3" in text
        assert "timeline gaps" in text
    else:
        raise AssertionError("Music final stream accepted a middle clip gap")


def test_music_final_waits_for_preview_gate_before_requesting_clip_latents():
    module = _load_stream_module()
    node = module.MiniMaxH3StreamLiveMusicVideoToVHS()
    common = dict(
        video_vae=object(), master_audio=object(), input_count=20,
        context_frames=39, video_overlap_frames=39,
        filename_prefix="x", pix_fmt="yuv420p", crf=19,
        save_metadata=False, trim_to_audio=False, save_output=True,
        active_clips=3,
        preview_gate=None,
        clip_1=None, clip_2=None, clip_3=None,
    )
    assert node.check_lazy_status(**common) == ["preview_gate"]
    common["preview_gate"] = ["preview-finished"]
    assert node.check_lazy_status(**common) == ["clip_1", "clip_2", "clip_3"]


def test_music_workflow_cache_isolates_active_clip_count_from_all_samplers():
    data = json.loads(WF.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {l[0]: l for l in data["links"]}
    controller = next(n for n in nodes.values() if n["type"] == "MiniMaxH3MusicVideoController")
    final = next(n for n in nodes.values() if n["type"] == "MiniMaxH3StreamLiveMusicVideoToVHS")
    barrier = next(n for n in nodes.values() if n["type"] == "MiniMaxH3LastActiveVHSPreviewBarrier")
    param = next(
        n for n in nodes.values()
        if n.get("properties", {}).get("h3_music_param") == "active_clips"
    )

    # The controller is frontend-only for execution selection. Changing its
    # widgets cannot become a backend dependency of any sampler/final node.
    assert not (controller["outputs"][0].get("links") or [])
    assert param["type"] == "PrimitiveInt"
    assert param["widgets_values"][0] == controller["widgets_values"][0] == 6

    cap_link = links[_input(final, "active_clips")["link"]]
    assert cap_link[1] == param["id"]
    barrier_cap = links[_input(barrier, "active_clips")["link"]]
    assert barrier_cap[1] == param["id"]

    param_dests = {links[lid][3] for lid in param["outputs"][0]["links"]}
    assert param_dests == {final["id"], barrier["id"]}
    assert not any(nodes[nid]["type"] == "SamplerCustomAdvanced" for nid in param_dests)


def test_music_workflow_has_explicit_last_active_preview_barrier_and_dynamic_40_clip_final():
    data = json.loads(WF.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {l[0]: l for l in data["links"]}
    final = next(n for n in nodes.values() if n["type"] == "MiniMaxH3StreamLiveMusicVideoToVHS")
    barrier = next(n for n in nodes.values() if n["type"] == "MiniMaxH3LastActiveVHSPreviewBarrier")

    assert final["widgets_values_named"]["input_count"] == 40
    assert final["widgets_values"][0] == 40
    assert barrier["widgets_values_named"]["input_count"] == 40
    assert barrier["widgets_values"][0] == 40
    assert not any(x["name"] == "preview_41" for x in barrier["inputs"])
    gate_link = links[_input(final, "preview_gate")["link"]]
    assert gate_link[1] == barrier["id"]

    for i in range(1, 41):
        clip = _input(final, f"clip_{i}")
        assert clip["link"] is not None
        preview = _input(barrier, f"preview_{i}")
        source = nodes[links[preview["link"]][1]]
        assert source["type"] == "VHS_VideoCombine"
        assert source.get("title") == f"Clip {i} - VHS Video Combine"


def test_music_controller_frontend_mirrors_only_active_count_into_internal_parameter():
    src = JS_CONTROLLER.read_text(encoding="utf-8")
    assert "h3_music_param" in src
    assert "syncExecutionParameters" in src
    assert "state.active" in src
    assert "expected exactly one cache-isolated Music Video parameter 'active_clips'" in src
    assert "const MAX_CLIPS = 40;" in src
    assert "state.active > managedClipCount" in src

    backend = (ROOT / "nodes.py").read_text(encoding="utf-8")
    controller_block = backend.split("class MiniMaxH3MusicVideoController:", 1)[1].split(
        "class MiniMaxH3AVExtensionController:", 1
    )[0]
    assert "MAX_CLIPS = 40" in controller_block

    dynamic = JS_INPUTS.read_text(encoding="utf-8")
    assert "MiniMaxH3StreamLiveMusicVideoToVHS" in dynamic
    assert 'prefix: "clip_"' in dynamic
    assert 'defaultInputs: 20' in dynamic
    assert '"Update inputs"' in dynamic
