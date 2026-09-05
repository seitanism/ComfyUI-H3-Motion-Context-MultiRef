import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "NEW - V2V Latent Motion Transfer (with upscale and de-rope).json"


def _load():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _node(data, node_id):
    return next(node for node in data["nodes"] if node["id"] == node_id)


def test_v2v_latent_motion_transfer_has_no_testing_vram_debug_node():
    data = _load()
    assert all(node["type"] != "VRAM_Debug" for node in data["nodes"])


def test_v2v_latent_motion_transfer_keeps_fractional_v2v_mask_semantics():
    data = _load()
    v2v = _node(data, 21)
    assert v2v["type"] == "H3V2VGranularFractionalDenoise"
    assert v2v["widgets_values_named"]["global_strength"] == 0.9995
    assert v2v["widgets_values_named"]["audio_strength"] == 0

    pass1_scheduler = _node(data, 33)
    pass2_scheduler = _node(data, 147)
    assert pass1_scheduler["type"] == "BasicScheduler"
    assert pass2_scheduler["type"] == "BasicScheduler"
    # Native fractional V2V strength belongs in the Pass-1 H3 mask, not scheduler denoise.
    assert pass1_scheduler["widgets_values_named"]["denoise"] == 1
    # Preserve the uploaded workflow's separate low-denoise Pass-2 refinement schedule.
    assert pass2_scheduler["widgets_values_named"]["denoise"] == 0.4


def test_v2v_latent_motion_transfer_keeps_two_pass_derope_upscale_chain():
    data = _load()
    assert _node(data, 34)["type"] == "SamplerCustomAdvanced"
    assert _node(data, 124)["type"] == "VAEDecode"
    assert _node(data, 123)["type"] == "H3JerkOracle"
    assert _node(data, 125)["type"] == "H3TimeSmear"
    assert _node(data, 154)["type"] == "MinimaxH3LatentUpscaler3D"
    assert _node(data, 156)["type"] == "MiniMaxChunkFeedForward"
    assert _node(data, 130)["type"] == "H3V2VInit"
    assert _node(data, 133)["type"] == "SamplerCustomAdvanced"
    assert _node(data, 135)["type"] == "H3ExactRecover"

    links = {link[0]: link for link in data["links"]}
    # Pass-1 sampler denoised/x0 output -> VAEDecode.
    assert links[28][1:5] == [34, 1, 124, 0]
    # Decoded pass-1 x0 -> generated-frame smear.
    assert links[30][1:5] == [124, 0, 125, 0]
    # Generated-frame smear -> VAEEncode -> learned latent upscaler.
    assert links[36][1:5] == [125, 0, 126, 0]
    assert links[89][1:5] == [126, 0, 154, 0]
    # Upscaled latent -> Pass-2 V2V init -> Pass-2 sampler.
    assert links[90][1:5] == [154, 0, 130, 0]
    assert links[88][1:5] == [130, 0, 133, 4]


def test_v2v_latent_motion_transfer_smears_source_only_for_pass2_reference():
    data = _load()
    pass1_ref = _node(data, 20)
    pass2_ref = _node(data, 131)
    source_smear = _node(data, 139)
    assert pass1_ref["type"] == "MiniMaxH3ReferenceToVideo"
    assert pass2_ref["type"] == "MiniMaxH3ReferenceToVideo"
    assert source_smear["type"] == "H3TimeSmear"

    links = {link[0]: link for link in data["links"]}
    # Original source passes 24fps validation, without time smearing in Pass 1.
    validator = next(n for n in data["nodes"] if n["type"] == "MiniMaxH3Validate24FPSVideo")
    source_link = next(i for i in validator["inputs"] if i["name"] == "images")["link"]
    assert links[source_link][1:3] == [1, 0]
    assert links[8][1:5] == [validator["id"], 0, 20, 6]
    # The same source is smeared only for Pass 2, using the generated hold map.
    assert links[32][1:5] == [validator["id"], 0, 139, 0]
    assert links[33][1:5] == [123, 0, 139, 1]
    assert links[47][1:5] == [139, 0, 131, 6]


def test_v2v_workflow_carries_current_usage_notes():
    data = _load()
    notes = {node["id"]: node for node in data["nodes"] if node["type"] == "Note"}
    assert {158, 159, 160, 161, 162}.issubset(notes)

    fractional = notes[161]["widgets_values_named"]["text"]
    assert "global_strength determines how much" in fractional
    assert "between 0.995 and 0.9997" in fractional

    streamed = notes[160]["widgets_values_named"]["text"]
    assert "almost 5 seconds" in streamed
    assert "increase the frame count 3 or 4 fold" in streamed

    pass2 = notes[162]["widgets_values_named"]["text"]
    assert "denoise 0.3 instead of the default 0.4" in pass2

    deps = notes[158]["widgets_values_named"]["text"]
    assert "ComfyUI-H3-Motion-Context-MultiRef (Update 8): H3V2VGranularFractionalDenoise" in deps
    assert "H3V2VNativeFractionalMaskDebugSource" not in deps
    assert "custom-node folder" not in deps.lower()
