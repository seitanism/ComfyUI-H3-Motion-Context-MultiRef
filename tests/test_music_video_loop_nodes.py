from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "NEW - Music Video (Compact Loop).json"


def _load(name, path):
    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(ROOT / "output")
        sys.modules["folder_paths"] = folder_paths
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clip_plan_is_a_native_output_list_and_preserves_order():
    module = _load("_h3_music_plan_test", ROOT / "h3_music_video_plan.py")
    cls = module.MiniMaxH3MusicVideoClipPlan
    assert cls.OUTPUT_IS_LIST == (True,)
    clips = cls().build(
        '[{"prompt":"first","seed":9},{"prompt":"second","seed":10}]'
    )[0]
    assert [(x["clip_number"], x["prompt"], x["seed"]) for x in clips] == [
        (1, "first", 9),
        (2, "second", 10),
    ]


def test_clip_plan_covers_audio_plus_five_seconds_and_repeats_shots():
    module = _load("_h3_music_plan_duration_test", ROOT / "h3_music_video_plan.py")

    class Waveform:
        ndim = 3
        shape = (1, 2, 60 * 32000)

    audio = {"sample_rate": 32000, "waveform": Waveform()}
    clips = module.MiniMaxH3MusicVideoClipPlan().build(
        '[{"prompt":"a","seed":1},{"prompt":"b","seed":2}]',
        audio,
        243,
        39,
        5.0,
    )[0]
    # 65 s => 1560 frames. 243 + 7 * (243 - 39) = 1671 frames.
    assert len(clips) == 8
    assert [clip["prompt"] for clip in clips] == ["a", "b"] * 4
    assert [clip["index"] for clip in clips] == list(range(8))
    assert [clip["total_clips"] for clip in clips] == [8] * 8


def test_current_clip_logs_a_high_visibility_progress_banner(caplog):
    import logging

    module = _load("_h3_music_plan_log_test", ROOT / "h3_music_video_plan.py")
    clips = module._parse_plan('[{"prompt":"a"},{"prompt":"b"},{"prompt":"c"}]')
    with caplog.at_level(logging.WARNING, logger="h3_motion_context.music_progress"):
        module.MiniMaxH3MusicVideoClipSpec().extract(clips[1], 243, 39)
    assert "MINIMAX H3 MUSIC VIDEO" in caplog.text
    assert "SEQUENCE 02 / 03" in caplog.text
    assert "=" * 20 in caplog.text


def test_current_clip_derives_overlap_aware_song_timeline():
    module = _load("_h3_music_plan_timing_test", ROOT / "h3_music_video_plan.py")
    clips = module._parse_plan('[{"prompt":"a"},{"prompt":"b"},{"prompt":"c"}]')
    node = module.MiniMaxH3MusicVideoClipSpec()
    assert node.extract(clips[0], 362, 39)[2] == 0.0
    assert node.extract(clips[1], 362, 39)[2] == pytest.approx(323 / 24)
    assert node.extract(clips[2], 362, 39)[2] == pytest.approx(646 / 24)


def test_current_clip_honors_explicit_start_and_rejects_bad_context():
    module = _load("_h3_music_plan_override_test", ROOT / "h3_music_video_plan.py")
    clip = module._parse_plan(
        '[{"prompt":"offset shot","seed":44,"start_seconds":7.25}]'
    )[0]
    assert module.MiniMaxH3MusicVideoClipSpec().extract(clip, 362, 39) == (
        "offset shot", 44, 7.25, 1
    )
    with pytest.raises(ValueError, match="context_frames"):
        module.MiniMaxH3MusicVideoClipSpec().extract(clip, 39, 39)


def test_plan_rejects_empty_or_invalid_clips():
    module = _load("_h3_music_plan_validation_test", ROOT / "h3_music_video_plan.py")
    for value in ("[]", '[{"seed":1}]', '[{"prompt":""}]'):
        with pytest.raises(ValueError):
            module._parse_plan(value)


def test_list_streamer_schema_is_loop_native():
    source = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    block = source.split("class MiniMaxH3StreamMusicVideoListToVHS:", 1)[1]
    assert '"clip_latents": ("LATENT", {"forceInput": True})' in block
    assert "INPUT_IS_LIST = True" in block
    assert "MiniMaxH3StreamLiveMusicVideoToVHS()" in block
    assert 'f"clip_{i}"' in block


def test_loop_nodes_are_registered():
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    for name in (
        "MiniMaxH3MusicVideoClipPlan",
        "MiniMaxH3MusicVideoClipSpec",
        "MiniMaxH3MusicVideoCheckpoint",
        "MiniMaxH3MusicVideoProjectName",
        "MiniMaxH3PreviewMusicVideoClipToVHS",
        "MiniMaxH3StreamMusicVideoListToVHS",
        "MiniMaxH3AssembleMusicVideoPreviews",
    ):
        assert source.count(f'"{name}"') >= 2


def test_compact_workflow_has_one_generation_body_and_no_per_clip_groups():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    types = [node["type"] for node in data["nodes"]]
    assert len(data["nodes"]) == 41
    assert len(data["groups"]) == 5
    assert types.count("StartLoop") == 1
    assert types.count("EndLoop") == 1
    assert types.count("MiniMaxH3ReferenceToVideo") == 1
    assert types.count("MiniMaxH3SongMaskedAVContext") == 1
    assert types.count("SamplerCustomAdvanced") == 1
    assert types.count("MiniMaxH3AssembleMusicVideoPreviews") == 1
    assert types.count("MiniMaxH3MusicVideoCheckpoint") == 1
    assert types.count("MiniMaxH3MusicVideoProjectName") == 1
    assert types.count("MiniMaxH3PreviewMusicVideoClipToVHS") == 1
    assert "MiniMaxH3PreviewOverride" not in types
    assert "MiniMaxH3AudioPreviewCS" not in types
    assert "MiniMaxH3MusicVideoController" not in types
    assert "MiniMaxH3LastActiveVHSPreviewBarrier" not in types
    assert "VHS_VideoCombine" not in types
    assert "ModelAttentionBackend" not in types
    assert not any(group["title"].startswith("CLIP ") for group in data["groups"])


def test_compact_workflow_defaults_to_ten_second_h3_clips():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    assert nodes[101]["widgets_values"] == [10]
    assert nodes[101]["widgets_values_named"]["value"] == 10
    assert nodes[100]["widgets_values"][1] == 0.4
    assert nodes[3002]["widgets_values"] == [243, 39]
    assert nodes[3000]["widgets_values"][1:] == [243, 39, 5.0]
    assert nodes[3004]["widgets_values"][:2] == [39, 5.0]


def test_compact_workflow_plan_is_driven_by_master_audio_duration():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    def source(node_id, input_name):
        item = next(i for i in nodes[node_id]["inputs"] if i["name"] == input_name)
        return links[item["link"]][1:3]

    assert source(3000, "master_audio") == [940, 0]
    assert source(3000, "frame_count") == [102, 1]
    assert source(3000, "context_frames") == [1733, 0]
    assert source(3004, "master_audio") == [940, 0]


def test_compact_workflow_contains_atos_four_minute_prompt_plan_and_references():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    plan = json.loads(nodes[3000]["widgets_values"][0])
    assert len(plan) == 29
    assert all("<Picture 1>" in item["prompt"] for item in plan)
    assert all("<Picture 2>" in item["prompt"] for item in plan)
    assert all(item["prompt"].startswith("r34l1sm\n") for item in plan)
    assert "On nous promet toujours demain" in plan[0]["prompt"]
    assert "FO ! FO ! FO !" in plan[-1]["prompt"]
    assert nodes[910]["widgets_values"][0] == "atos_musician_portrait.png"
    assert nodes[911]["widgets_values"][0] == "atos_musician_full.png"
    assert nodes[940]["widgets_values"][0] == "FO Pouvoir d'achat.mp3"
    assert nodes[940]["widgets_values_named"]["audio"] == "FO Pouvoir d'achat.mp3"
    assert nodes[3005]["widgets_values"] == [1, "atos_fo_4min"]
    assert nodes[3006]["widgets_values"] == [True, "h3_preview/atos_fo_4min", True]


def test_compact_workflow_loop_carries_previous_latent_and_accumulates_all_clips():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    def source(node_id, input_name):
        item = next(i for i in nodes[node_id]["inputs"] if i["name"] == input_name)
        return links[item["link"]][1:3]

    assert source(3001, "list") == [3000, 0]
    assert source(3002, "clip") == [3001, 3]
    assert source(1755, "source_latent") == [3001, 4]
    assert source(1755, "is_last_clip") == [3001, 2]
    assert source(3005, "generated_latent") == [124, 0]
    assert source(3005, "clip_number") == [3002, 3]
    assert source(3006, "clip_latent") == [3005, 0]
    assert source(3006, "clip_audio") == [1755, 2]
    assert source(3006, "clip_number") == [3002, 3]
    assert source(3006, "resume_from") == [3005, 1]
    assert source(3003, "output_value") == [3002, 3]
    assert source(3003, "next_iteration_value") == [3006, 0]
    assert nodes[3003]["widgets_values"] == [True]
    assert source(3004, "completed_clips") == [3003, 0]
    assert source(3005, "project_name") == [3020, 0]
    assert source(3006, "project_name") == [3020, 0]
    assert source(3004, "project_name") == [3020, 0]


def test_checkpoint_lazily_skips_prefix_and_round_trips_h3_latent(tmp_path):
    import torch

    module = _load("_h3_music_checkpoint_test", ROOT / "h3_music_video_checkpoint.py")
    module.folder_paths.get_output_directory = lambda: str(tmp_path)
    node = module.MiniMaxH3MusicVideoCheckpoint()
    latent = {
        "samples": [
            torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1),
            torch.arange(4, dtype=torch.float32).reshape(1, 1, 4),
        ]
    }

    assert node.check_lazy_status(None, 1, 1, "take_a") == ["generated_latent"]
    saved = node.checkpoint(latent, 1, 1, "take_a")
    assert saved[0] is latent
    assert saved[1] == 1
    assert node.check_lazy_status(None, 1, 5, "take_a") == []
    loaded = node.checkpoint(None, 1, 5, "take_a")[0]
    assert torch.equal(loaded["samples"][0], latent["samples"][0])
    assert torch.equal(loaded["samples"][1], latent["samples"][1])


def test_clip_preview_skips_disabled_and_restored_prefix_without_decoding():
    import inspect

    module = _load("_h3_music_preview_test", ROOT / "h3_streaming_vhs.py")
    signature = inspect.signature(module.MiniMaxH3StreamLiveMusicVideoToVHS.stream_to_vhs)
    assert signature.parameters["tail_seconds"].default == 0.0
    node = module.MiniMaxH3PreviewMusicVideoClipToVHS()
    latent = {"samples": "sentinel"}
    assert node.preview_clip(latent, 5, 1, enabled=False) == (latent,)
    assert node.preview_clip(latent, 4, 5, enabled=True) == (latent,)
    assert node.check_lazy_status(latent, 4, 5, enabled=True) == []
    assert node.check_lazy_status(latent, 5, 5, enabled=True) == ["video_vae", "clip_audio"]


def test_preview_assembler_consumes_the_completed_clip_list_as_one_input():
    module = _load("_h3_preview_assembler_test", ROOT / "h3_preview_assembler.py")
    node = module.MiniMaxH3AssembleMusicVideoPreviews()
    assert node.INPUT_IS_LIST is True
    assert node._clip_count([1, 2, 3, 4]) == 4
    assert node._one([39], "context_frames") == 39


def test_compact_workflow_optimization_chain_ends_with_veda():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    def source(node_id, input_name="model"):
        item = next(i for i in nodes[node_id]["inputs"] if i["name"] == input_name)
        return links[item["link"]][1]

    expected = [
        (977, "LoraLoaderModelOnly", 1),
        (3019, "LoraLoaderModelOnly", 977),
        (3010, "MiniMaxH3MotionCache", 3019),
        (3011, "MiniMaxH3MemoryEfficientSageAttentionPatch", 3010),
        (3012, "PathchSageAttentionKJ", 3011),
        (3013, "MiniMaxChunkFeedForward", 3012),
        (3014, "ModelPatchTorchSettings", 3013),
        (3015, "SolAttnPatch", 3014),
        (3016, "VedaSparseAttention", 3015),
        (5, "MiniMaxH3SigmaShift", 3016),
    ]
    for node_id, node_type, source_id in expected:
        assert nodes[node_id]["type"] == node_type
        assert source(node_id) == source_id
    assert nodes[3016]["title"] == "7 — VEDA (MUST BE LAST)"
    assert nodes[3016]["widgets_values"][0] == "minimax_h3_r2va_veda_preview_fp8.safetensors"
    assert nodes[3019]["widgets_values"] == [
        "h3-realism-people-t2v-i2v-r2v.safetensors", 0.7
    ]

    assert 3017 not in nodes
    assert 3018 not in nodes
    assert source(121) == 5
    assert source(123) == 5


def test_compact_workflow_links_are_consistent():
    import json

    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    for node in nodes.values():
        for slot, inp in enumerate(node.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is not None:
                assert links[link_id][3:5] == [node["id"], slot]
        for slot, out in enumerate(node.get("outputs", [])):
            for link_id in out.get("links") or []:
                assert links[link_id][1:3] == [node["id"], slot]
