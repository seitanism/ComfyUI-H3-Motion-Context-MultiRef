"""Static checks for the Update 3 masked example workflows."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"


def _load(name):
    data = json.loads((WF / name).read_text(encoding="utf-8"))
    ids = {n["id"] for n in data["nodes"]}
    link_ids = {link[0] for link in data["links"]}
    assert len(ids) == len(data["nodes"])
    for link in data["links"]:
        assert link[1] in ids
        assert link[3] in ids
    for node in data["nodes"]:
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is not None:
                assert lid in link_ids
        for out in node.get("outputs", []):
            for lid in out.get("links") or []:
                assert lid in link_ids
    return data


def _types(data):
    return [n["type"] for n in data["nodes"]]


def _node(data, type_name):
    return next(n for n in data["nodes"] if n["type"] == type_name)



def test_masked_two_video_bridge_example():
    data = _load("UTILITY - AV Bridge.json")
    types = _types(data)
    assert "MiniMaxH3MaskedAVBridge" in types
    assert "MiniMaxH3AddGuide" not in types

    by_id = {n["id"]: n for n in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    timing = _node(data, "MiniMaxH3AVBridgeTiming")
    assert timing["widgets_values"] == [192, 39]
    assert "PrimitiveFloat" not in types
    assert "TrimAudioDuration" not in types
    assert "AudioConcat" not in types
    target = _node(data, "MiniMaxH3ImageToVideo")
    target_link = next(i for i in target["inputs"] if i["name"] == "length")["link"]
    assert links[target_link][1:3] == [timing["id"], 0]
    bridge = _node(data, "MiniMaxH3MaskedAVBridge")
    for name in ("start_fps", "end_fps"):
        assert next(i for i in bridge["inputs"] if i["name"] == name)["link"] is None
    assert bridge["widgets_values"][:3] == [24.0, 24.0, 39]
    preserve_link = next(i for i in bridge["inputs"] if i["name"] == "preserve_frames")["link"]
    assert links[preserve_link][1:3] == [timing["id"], 1]
    audio = _node(data, "MiniMaxH3AssembleBridgeAudio")
    for name, slot in (("target_frames", 0), ("preserve_frames", 1)):
        lid = next(i for i in audio["inputs"] if i["name"] == name)["link"]
        assert links[lid][1:3] == [timing["id"], slot]
    for stitch in [n for n in data["nodes"] if n["type"] == "ImageBatchExtendWithOverlap"]:
        lid = next(i for i in stitch["inputs"] if i["name"] == "overlap")["link"]
        assert links[lid][1:3] == [bridge["id"], 2]
    output = _node(data, "CreateVideo")
    assert output["widgets_values"] == [24.0]
    lid = next(i for i in output["inputs"] if i["name"] == "audio")["link"]
    assert links[lid][1:3] == [audio["id"], 0]
    assert types.count("MiniMaxH3Validate24FPSVideo") == 2
