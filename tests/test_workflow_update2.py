import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"

SIMPLE = "OLD - Motion Context - Simple.json"
ADVANCED = "OLD - Motion Context - Advanced.json"
MP4 = "OLD - Hybrid Extension.json"
CUSTOM = "UTILITY - Custom Keyframes.json"


def _load(name):
    data = json.loads((WF / name).read_text(encoding="utf-8"))
    return data, {n["id"]: n for n in data["nodes"]}, {l[0]: l for l in data["links"]}


def _by_title(wf, title):
    return [n for n in wf["nodes"] if n.get("title") == title]


def _nodes_of(wf, node_type):
    return [n for n in wf["nodes"] if n.get("type") == node_type]


def _input(node, name):
    return next(i for i in node.get("inputs", []) if i.get("name") == name)


def _assert_common_continuation_workflow(name, expected_context_nodes):
    wf, nodes, links = _load(name)

    context_globals = _by_title(wf, "GLOBAL CONTEXT FRAMES")
    xfade_globals = _by_title(wf, "GLOBAL VIDEO CROSSFADE")
    assert len(context_globals) == 1
    assert len(xfade_globals) == 1
    assert context_globals[0]["widgets_values"][0] == 39
    assert xfade_globals[0]["widgets_values"][0] == 39

    guide_notes = [
        n for n in wf["nodes"]
        if n.get("title") in {"Context length guide", "CONTEXT LENGTH GUIDE"}
    ]
    assert len(guide_notes) == 1
    note = guide_notes[0]["widgets_values"][0]
    for value in ("5", "22", "39*", "56", "73", "90*", "141*", "192*", "243*"):
        assert value in note
    assert "exact video+audio boundary" in note
    assert "39 recommended" in note

    mc_nodes = _nodes_of(wf, "MiniMaxH3MotionContext")
    assert len(mc_nodes) == expected_context_nodes
    context_global_id = context_globals[0]["id"]
    for node in mc_nodes:
        inp = _input(node, "context_length")
        assert inp["link"] is not None
        assert links[inp["link"]][1] == context_global_id
        assert node["widgets_values"][0] == 39

    kj_nodes = _nodes_of(wf, "ImageBatchExtendWithOverlap")
    assert len(kj_nodes) == expected_context_nodes
    for node in kj_nodes:
        assert node["widgets_values"] == [39, "source", "linear_blend"]
        assert _input(node, "overlap")["link"] is not None
        # Update 2 drives the actual overlap from Trim's clamped output so
        # context=90 / crossfade=39 cannot leave duplicated prefix frames.
        overlap_link = links[_input(node, "overlap")["link"]]
        upstream = nodes[overlap_link[1]]
        assert upstream["type"] == "MiniMaxH3MotionContextTrim"
        assert overlap_link[2] == 3

    # Every continuation Trim used by a KJ blend exposes the new retained-video
    # outputs while preserving output 0/1 for backward-compatible hard trimming.
    trim_ids = {links[_input(kj, "overlap")["link"]][1] for kj in kj_nodes}
    for trim_id in trim_ids:
        trim = nodes[trim_id]
        assert [o["name"] for o in trim["outputs"]] == [
            "images", "audio", "crossfade_images", "crossfade_frames"
        ]
        assert _input(trim, "video_crossfade_frames")["link"] is not None

    return wf, nodes, links


def test_simple_context_audio_and_crossfade():
    wf, nodes, links = _assert_common_continuation_workflow(SIMPLE, 5)
    context_global = _by_title(wf, "GLOBAL CONTEXT FRAMES")[0]
    for mc in _nodes_of(wf, "MiniMaxH3MotionContext"):
        audio_in = _input(mc, "audio_context_length")
        assert audio_in["link"] is not None
        assert links[audio_in["link"]][1] == context_global["id"]
        assert mc["widgets_values"][4] == 39


def test_advanced_context_audio_and_crossfade():
    wf, nodes, links = _assert_common_continuation_workflow(ADVANCED, 5)
    context_global = _by_title(wf, "GLOBAL CONTEXT FRAMES")[0]
    for mc in _nodes_of(wf, "MiniMaxH3MotionContext"):
        audio_in = _input(mc, "audio_context_length")
        assert audio_in["link"] is not None
        assert links[audio_in["link"]][1] == context_global["id"]
        assert mc["widgets_values"][4] == 39


def test_existing_mp4_workflow_uses_masked_prefix_and_kj_crossfade():
    wf, nodes, links = _load(MP4)
    context_globals = _by_title(wf, "GLOBAL CONTEXT FRAMES")
    xfade_globals = _by_title(wf, "GLOBAL VIDEO CROSSFADE")
    assert len(context_globals) == len(xfade_globals) == 1
    assert context_globals[0]["widgets_values"][0] == 39
    assert xfade_globals[0]["widgets_values"][0] == 39

    masked = _nodes_of(wf, "MiniMaxH3ExistingVideoMaskedContext")
    assert len(masked) == 1
    ctx_input = _input(masked[0], "context_length")
    assert links[ctx_input["link"]][1] == context_globals[0]["id"]

    loaders = _nodes_of(wf, "VHS_LoadVideo")
    assert len(loaders) == 1
    assert loaders[0]["widgets_values"]["force_rate"] == 24

    kj = _nodes_of(wf, "ImageBatchExtendWithOverlap")
    assert len(kj) == 6
    for node in kj:
        assert node["widgets_values"] == [39, "source", "linear_blend"]
        overlap_link = links[_input(node, "overlap")["link"]]
        trim = nodes[overlap_link[1]]
        assert trim["type"] == "MiniMaxH3MotionContextTrim"
        assert overlap_link[2] == 3

    guide = [n for n in wf["nodes"] if n.get("title") in {"Context length guide", "CONTEXT LENGTH GUIDE"}]
    assert len(guide) == 1
    text = guide[0]["widgets_values"][0]
    assert "39*" in text and "90*" in text and "141*" in text and "192*" in text


def test_custom_keyframe_example_is_not_update2_continuation_graph():
    wf, _, _ = _load(CUSTOM)
    assert not _nodes_of(wf, "ImageBatchExtendWithOverlap")
    assert not _nodes_of(wf, "MiniMaxH3ExistingVideoMaskedContext")
    assert not _by_title(wf, "GLOBAL CONTEXT FRAMES")
