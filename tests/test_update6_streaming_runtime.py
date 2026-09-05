from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_direct_streaming_runtime_is_registered_and_experimental_paths_are_absent():
    nodes = (ROOT / "nodes.py").read_text(encoding="utf-8")
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    assert '"MiniMaxH3StreamLiveExtensionAVToVHS"' in nodes
    assert '"MiniMaxH3StreamLiveMusicVideoToVHS"' in nodes
    assert '"MiniMaxH3FinalizeVHSOutput"' in nodes
    assert "class MiniMaxH3StreamLiveExtensionAVToVHS" in stream
    assert "class MiniMaxH3StreamLiveMusicVideoToVHS" in stream
    assert "class MiniMaxH3FinalizeVHSOutput" in stream
    assert "MiniMaxH3LowRamPreviewToVHS" not in stream
    assert "VHS_BatchManager" not in stream
    assert "Meta Batch" not in stream
    # Superseded materialized final-movie assemblers must not remain registered
    # or available as alternative nodes: they recreated the large final RGB buffer.
    assert "MiniMaxH3AssembleLiveExtensionAV" not in nodes
    assert "MiniMaxH3AssembleLiveMusicVideo" not in nodes
    assert "class MiniMaxH3AssembleLiveExtensionAV" not in (ROOT / "existing_video_extension.py").read_text(encoding="utf-8")
    assert "class MiniMaxH3AssembleLiveMusicVideo" not in (ROOT / "h3_song_audio_context.py").read_text(encoding="utf-8")


def test_streaming_runtime_uses_one_shot_frames_and_exact_audio_conformance():
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    assert "class _OneShotFrameSequence" in stream
    assert "old final RGB buffer" in stream
    assert "_conform_waveform_length" in stream
    assert "extension_start_frame = cumulative_frames - int(contexts[i])" in stream
    assert "extension_start_sample = sample_boundary_from_frames(" in stream
    assert "extension_end_sample = sample_boundary_from_frames(" in stream
    assert "audio_out[..., extension_start_sample:extension_end_sample].copy_(" in stream
    assert "wave = wave[..., cut:]" not in stream


def test_music_stream_is_intermediate_and_terminal_sink_is_output_node():
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    music = stream.split("class MiniMaxH3StreamLiveMusicVideoToVHS:", 1)[1]
    music = music.split("class ", 1)[0] if "class " in music else music
    assert "OUTPUT_NODE = False" in music
    assert "HAS_INTERMEDIATE_OUTPUT = True" in music
    sink = stream.split("class MiniMaxH3FinalizeVHSOutput:", 1)[1].split("class MiniMaxH3StreamLiveMusicVideoToVHS:", 1)[0]
    assert "OUTPUT_NODE = True" in sink


def test_final_output_filename_allocation_stays_owned_by_vhs():
    """Guard against reintroducing fixed-path final-video overwrites."""
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    wrapper = stream.split("def _run_vhs_h264(", 1)[1].split("\n\nclass MiniMaxH3StreamLiveExtensionAVToVHS", 1)[0]

    # The H3 wrapper must hand the user prefix to stock VHS and let VHS choose
    # the numbered output path.  It must not become a custom file writer again.
    assert "vhs.combine_video(" in wrapper
    assert "filename_prefix=_expand_output_prefix(filename_prefix)" in wrapper
    for destructive in (
        "os.remove(", "os.unlink(", "os.replace(", "os.rename(",
        "Path.unlink(", "subprocess.", "ffmpeg",
    ):
        assert destructive not in wrapper


def test_new_workflows_save_final_outputs_with_stable_vhs_prefixes():
    import json

    cases = [
        ("NEW - AV Extension.json", "MiniMaxH3StreamLiveExtensionAVToVHS", "video/masked_av_extension"),
        ("NEW - Music Video.json", "MiniMaxH3StreamLiveMusicVideoToVHS", "video/h3_music_video"),
    ]
    for filename, node_type, expected_prefix in cases:
        workflow = json.loads((ROOT / "example_workflows" / filename).read_text(encoding="utf-8"))
        final = next(node for node in workflow["nodes"] if node["type"] == node_type)
        values = final["widgets_values_named"]
        assert values["save_output"] is True
        assert values["filename_prefix"] == expected_prefix
        # Keep VHS's normal output bookkeeping enabled in the shipped examples.
        assert workflow.get("extra", {}).get("VHS_KeepIntermediate", True) is True
