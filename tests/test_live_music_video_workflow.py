import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows" / "NEW - Music Video.json"


def load():
    return json.loads(WF.read_text(encoding="utf-8"))


def _nodes(data):
    return {n["id"]: n for n in data["nodes"]}


def _links(data):
    return {l[0]: l for l in data["links"]}


def _input(node, name):
    return next(i for i in node.get("inputs", []) if i["name"] == name)


def test_live_music_video_is_checkpoint_free_40_clip_ref2va_chain():
    data = load()
    types = [n["type"] for n in data["nodes"]]
    assert types.count("MiniMaxH3ReferenceToVideo") == 40
    assert types.count("MiniMaxH3SongMaskedAVContext") == 40
    assert types.count("SamplerCustomAdvanced") == 40
    assert types.count("MiniMaxH3MusicVideoController") == 1
    assert types.count("MiniMaxH3StreamLiveMusicVideoToVHS") == 1
    assert types.count("MiniMaxH3FinalizeVHSOutput") == 1
    assert types.count("MiniMaxH3AssembleLiveMusicVideo") == 0
    assert types.count("VHS_VideoCombine") == 40  # clip previews only; no separate final VHS node
    assert not any("Checkpoint" in t for t in types)
    assert not any("rgthree" in t.lower() for t in types)
    loader = next(n for n in data["nodes"] if n["type"] == "UNETLoader")
    assert loader["widgets_values"][0] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def test_music_controller_defaults_and_group_ownership():
    data = load(); owners = defaultdict(list)
    controller = next(n for n in data["nodes"] if n["type"] == "MiniMaxH3MusicVideoController")
    assert controller["widgets_values"] == [6, "All Active"]
    managed = []
    for g in data.get("groups", []):
        meta = (g.get("flags") or {}).get("h3_control")
        if not meta:
            continue
        b = g["bounding"]; members = []
        for n in data["nodes"]:
            cx = n["pos"][0] + n["size"][0] / 2
            cy = n["pos"][1] + n["size"][1] / 2
            if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]:
                members.append(n); owners[n["id"]].append(meta)
        managed.append((meta, members))
    assert not {nid: v for nid, v in owners.items() if len(v) > 1}
    assert sum(m["role"] == "music_clip" for m, _ in managed) == 40
    assert sum(m["role"] == "music_preview" for m, _ in managed) == 40
    assert {(m["role"], m["index"]) for m, _ in managed} == {
        (role, index)
        for role in ("music_clip", "music_preview")
        for index in range(1, 41)
    }
    for meta, members in managed:
        enabled = meta.get("index") <= 6
        assert all(n.get("mode", 0) == (0 if enabled else 4) for n in members)


def test_four_global_images_feed_all_forty_ref2va_nodes():
    data = load(); nodes = _nodes(data); links = _links(data)
    # Identify the four global references by graph type/role rather than optional display titles.
    refs = [n for n in nodes.values() if n["type"] == "LoadImage"]
    assert len(refs) == 4
    assert {n["id"] for n in refs} == {910, 911, 2505, 2506}
    active = [n for n in refs if n.get("mode", 0) == 0]
    bypass = [n for n in refs if n.get("mode", 0) == 4]
    assert len(active) == 2 and len(bypass) == 2
    for n in nodes.values():
        if n["type"] != "MiniMaxH3ReferenceToVideo":
            continue
        sockets = [_input(n, f"ref_images.ref_image_{i}") for i in range(4)]
        assert all(s.get("link") is not None for s in sockets)
        assert {links[s["link"]][1] for s in sockets} == {r["id"] for r in refs}


def test_each_continuation_uses_previous_sampler_latent_directly():
    data = load(); nodes = _nodes(data); links = _links(data)
    final = next(n for n in nodes.values() if n["type"] == "MiniMaxH3StreamLiveMusicVideoToVHS")
    samplers = []
    contexts = []
    for i in range(1, 41):
        clip_link = _input(final, f"clip_{i}")["link"]
        assert clip_link is not None
        sampler = nodes[links[clip_link][1]]
        assert sampler["type"] == "SamplerCustomAdvanced"
        samplers.append(sampler)

        latent_link = _input(sampler, "latent_image")["link"]
        assert latent_link is not None
        context = nodes[links[latent_link][1]]
        assert context["type"] == "MiniMaxH3SongMaskedAVContext"
        contexts.append(context)

        start_link = _input(context, "clip_start_seconds")["link"]
        start = nodes[links[start_link][1]]
        assert start["type"] == "ComfyMathExpression"
        assert start["widgets_values_named"]["expression"] == f"a * {i - 1}"

    assert len(samplers) == len(contexts) == 40
    assert _input(contexts[0], "source_latent")["link"] is None
    for i in range(1, 40):
        source_link = _input(contexts[i], "source_latent")["link"]
        assert source_link is not None
        source = links[source_link][1]
        assert source == samplers[i - 1]["id"]


def test_master_song_timing_and_final_output_are_live_and_untrimmed():
    data = load(); nodes = _nodes(data); links = _links(data)
    master = next(n for n in nodes.values() if n["type"] == "LoadAudio")
    assert master["widgets_values"][0] == "I\'ll Know You by the Scar.wav"
    final = next(n for n in nodes.values() if n["type"] == "MiniMaxH3StreamLiveMusicVideoToVHS")
    for i in range(1, 41):
        assert _input(final, f"clip_{i}")["link"] is not None
    assert links[_input(final, "master_audio")["link"]][1] == master["id"]
    assert final["outputs"][0]["type"] == "VHS_FILENAMES"
    sink = next(n for n in nodes.values() if n["type"] == "MiniMaxH3FinalizeVHSOutput")
    sink_link = _input(sink, "filenames")["link"]
    assert links[sink_link][1] == final["id"]
    assert final["widgets_values_named"]["trim_to_audio"] is False
    assert final["widgets_values_named"]["save_output"] is True
    assert not any(n.get("title") == "Final Output — VHS Video Combine" for n in nodes.values())


def test_director_prompt_uses_frame_derived_song_slice_math():
    data = load()
    director = next(n for n in data["nodes"] if n.get("id") == 1732)
    text = director["widgets_values"][0]
    assert "requested_frames = max(5, round(raw_clip_duration_seconds × 24))" in text
    assert "timeline_advance_frames = h3_frame_count - context_frames" in text
    assert "clip_start_frame = (N - 1) × timeline_advance_frames" in text
    assert "clip_end_frame = clip_start_frame + h3_frame_count" in text
    assert "exact frame-derived clip start/end boundaries" in text


def test_music_workflow_links_are_internally_consistent():
    data = load(); nodes = _nodes(data); links = _links(data)
    for n in nodes.values():
        for slot, inp in enumerate(n.get("inputs", [])):
            lid = inp.get("link")
            if lid is None:
                continue
            assert lid in links and links[lid][3] == n["id"] and links[lid][4] == slot
        for slot, out in enumerate(n.get("outputs", [])):
            for lid in out.get("links") or []:
                assert lid in links and links[lid][1] == n["id"] and links[lid][2] == slot


def test_music_controller_frontend_uses_real_bypass_and_metadata_groups():
    src = (ROOT / "js" / "h3_music_video_controller.js").read_text(encoding="utf-8")
    assert "MODE_BYPASS = 4" in src
    assert "group?.flags?.h3_control" in src
    assert "beforeQueued" in src
    assert "afterConfigureGraph" in src
    assert "setInterval" in src
    assert "nativeWouldSkip" in src
    assert "this.updateParameters(params, true)" in src
    assert ".updateSource(" not in src


def test_bundled_six_clip_demo_settings_and_assets_match_original_example():
    data = load(); nodes = _nodes(data)
    assert nodes[910]["widgets_values"][0] == "be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png"
    assert nodes[911]["widgets_values"][0] == "c90ee577-98eb-4f6c-9b0c-562a6b448d69.png"
    assert nodes[2505]["widgets_values"][0] == "" and nodes[2505]["mode"] == 4
    assert nodes[2506]["widgets_values"][0] == "" and nodes[2506]["mode"] == 4
    assert nodes[100]["widgets_values"] == ["16:9 (Widescreen)", 1, 32]
    assert nodes[101]["widgets_values"] == [15]
    assert nodes[970]["widgets_values"] == [8, "fixed"]
    assert nodes[1733]["widgets_values"] == [39, "fixed"]
    assert nodes[1734]["widgets_values"] == [39, "fixed"]
    assert nodes[973]["widgets_values"] == ["res_multistep"]
    assert nodes[123]["widgets_values"] == ["simple", 20, 1]
    assert nodes[977]["widgets_values"] == [
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1
    ]

    seed_ids = [
        120, 210, 310, 410, 510, 610, 710, 1010, 1110, 1210,
        1310, 1410, 1510, 1610, 1710, 1810, 1910, 2010, 2110, 2210,
    ]
    expected = [
        514005817509111, 903826866713850, 208140829245950, 378941378675234,
        162450393808085, 661271193620413, 889909296941847, 587177660189550,
        907464430774817, 866168170608469, 120909092658224, 679731235512457,
        490884057795672, 534836803595541, 899262098164995, 160016001600160,
        161016101610161, 162016201620162, 163016301630163, 164016401640164,
    ]
    assert [nodes[nid]["widgets_values"][0] for nid in seed_ids] == expected
    assert all(nodes[nid]["widgets_values"][1] == "fixed" for nid in seed_ids)

    # Lock the corrected bundled Clip 1-6 prompts too. These hashes intentionally
    # differ from the older example because the stale <Audio 1> wording was removed.
    prompt_ids = [110, 200, 300, 400, 500, 600]
    prompt_hashes = [
        "f0ad55152380122fee6936ae1d0b2c3784ae7f5725097be10cec03ac5f6cf7d7",
        "02c682af0295f84d4af7c09e497905c95cd3a74184cef7509dce7da220ebab2f",
        "54b74340d8af2a22ebc4aa252f479a15c4f1032877c7820eafa698600b343007",
        "3ef8610c78dcac9c0c6a6392fc4ece9afaa174e14e4f86ecc419d123d68dd1f5",
        "fd9f074709de904caf6fd1f56903566c69d85b52ecff128d4fe66ccabdaff3d4",
        "7755075a369cbeb9e37a6fa5e79d16d50ab208b40cda2ac05b775b395bb8b10e",
    ]
    actual_hashes = []
    for nid in prompt_ids:
        prompt = nodes[nid].get("widgets_values_named", {}).get("prompt", nodes[nid]["widgets_values"][0])
        actual_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    assert actual_hashes == prompt_hashes

    assets = ROOT / "example_workflows" / "assets"
    for name in [
        "be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png",
        "c90ee577-98eb-4f6c-9b0c-562a6b448d69.png",
        "I'll Know You by the Scar.wav",
        "lyrics.txt",
        "LICENSE.md",
    ]:
        assert (assets / name).is_file()


def test_bundled_demo_keeps_song_out_of_ref_audio_sockets():
    data = load()
    for n in data["nodes"]:
        if n["type"] != "MiniMaxH3ReferenceToVideo":
            continue
        for inp in n.get("inputs", []):
            if inp["name"].startswith("ref_audios.ref_audio_"):
                assert inp.get("link") is None


def test_music_preview_barrier_orders_last_active_preview_before_final_stream():
    data = load(); nodes = _nodes(data); links = _links(data)
    final = next(n for n in nodes.values() if n["type"] == "MiniMaxH3StreamLiveMusicVideoToVHS")
    barrier = next(n for n in nodes.values() if n["type"] == "MiniMaxH3LastActiveVHSPreviewBarrier")
    sink = next(n for n in nodes.values() if n["type"] == "MiniMaxH3FinalizeVHSOutput")
    assert links[_input(sink, "filenames")["link"]][1] == final["id"]
    assert links[_input(final, "preview_gate")["link"]][1] == barrier["id"]

    stream_src = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    music_block = stream_src.split("class MiniMaxH3StreamLiveMusicVideoToVHS:", 1)[1]
    assert "OUTPUT_NODE = False" in music_block
    assert 'if "preview_gate" in kwargs and kwargs["preview_gate"] is None:' in music_block
    sink_block = stream_src.split("class MiniMaxH3FinalizeVHSOutput:", 1)[1].split("class MiniMaxH3StreamLiveMusicVideoToVHS:", 1)[0]
    assert "OUTPUT_NODE = True" in sink_block

    previews = [n for n in nodes.values() if n["type"] == "VHS_VideoCombine"]
    for i in range(1, 41):
        preview_link = links[_input(barrier, f"preview_{i}")["link"]]
        preview = nodes[preview_link[1]]
        assert preview["type"] == "VHS_VideoCombine"
        assert preview.get("title") == f"Clip {i} - VHS Video Combine"
        assert preview["widgets_values_named"]["filename_prefix"] == f"h3_preview/music_clip_{i:02d}"

    # For active demo clips, prove each preview decode comes from the same sampler
    # supplied to the final stream; do not rely on optional/custom titles.
    for i in range(1, 7):
        clip_link = _input(final, f"clip_{i}")["link"]
        assert clip_link is not None
        sampler = nodes[links[clip_link][1]]
        matching_previews = []
        for preview in previews:
            images_link = _input(preview, "images")["link"]
            if images_link is None:
                continue
            decode = nodes[links[images_link][1]]
            if decode["type"] != "VAEDecode":
                continue
            samples_link = _input(decode, "samples")["link"]
            if samples_link is not None and links[samples_link][1] == sampler["id"]:
                matching_previews.append(preview)
        assert len(matching_previews) == 1


def test_bundled_six_clip_demo_prompts_do_not_invent_ref_audio_label():
    data = load()
    for nid in [110, 200, 300, 400, 500, 600]:
        node = next(n for n in data["nodes"] if n["id"] == nid)
        prompt = node.get("widgets_values_named", {}).get("prompt", "")
        assert "<Audio 1>" not in prompt
        assert "[reference generation + audio reference]" not in prompt
        assert "[reference generation]" in prompt
