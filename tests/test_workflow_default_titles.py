import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF_DIR = ROOT / "example_workflows"

DEFAULT_SUFFIX = {
    "MiniMaxH3ReferenceToVideo": "MiniMax H3 Reference to Video",
    "SamplerCustomAdvanced": "SamplerCustomAdvanced",
    "VHS_VideoCombine": "VHS Video Combine",
    "UNETLoader": "Load Diffusion Model",
    "CLIPLoader": "Load CLIP",
    "VAELoader": "Load VAE",
    "LoraLoaderModelOnly": "LoraLoaderModelOnly",
}
LOADER_TYPES = {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly"}
PREFIXED_TYPES = set(DEFAULT_SUFFIX)


def _load(name):
    return json.loads((WF_DIR / name).read_text(encoding="utf-8"))


def test_utility_workflows_use_only_default_node_titles():
    for name in ["UTILITY - AV Bridge.json", "UTILITY - Custom Keyframes.json"]:
        data = _load(name)
        assert all("title" not in node for node in data["nodes"])


def test_new_workflows_only_prefix_prompt_sampler_vhs_and_model_loaders():
    av = _load("NEW - AV Extension.json")
    music = _load("NEW - Music Video.json")

    for node in av["nodes"]:
        if node["type"] not in PREFIXED_TYPES:
            assert "title" not in node
            continue
        title = node.get("title")
        assert title is not None
        if node["type"] in LOADER_TYPES:
            assert title == f"Starter Clip - {DEFAULT_SUFFIX[node['type']]}"
        else:
            assert title.endswith(f" - {DEFAULT_SUFFIX[node['type']]}")
            prefix = title[: -len(f" - {DEFAULT_SUFFIX[node['type']]}")]
            assert (
                prefix == "Starter Clip"
                or prefix == "Source Audio Regen"
                or prefix in {f"Extension {i}" for i in range(1, 7)}
            )

    for node in music["nodes"]:
        if node["type"] not in PREFIXED_TYPES:
            assert "title" not in node
            continue
        title = node.get("title")
        assert title is not None
        if node["type"] in LOADER_TYPES:
            assert title == f"Clip 1 - {DEFAULT_SUFFIX[node['type']]}"
        else:
            assert title.endswith(f" - {DEFAULT_SUFFIX[node['type']]}")
            prefix = title[: -len(f" - {DEFAULT_SUFFIX[node['type']]}")]
            assert prefix in {f"Clip {i}" for i in range(1, 41)}

def test_v2v_latent_motion_transfer_only_prefixes_stage_defining_nodes():
    data = _load("NEW - V2V Latent Motion Transfer (with upscale and de-rope).json")
    expected = {
        10: "PASS 1 - Load Diffusion Model",
        20: "PASS 1 - MiniMax H3 Reference to Video",
        21: "PASS 1 - H3 V2V Granular Fractional Denoise",
        34: "PASS 1 - SamplerCustomAdvanced",
        149: "PASS 2 - Load Diffusion Model",
        156: "Pass 2 — Chunk Feed-Forward",
        131: "PASS 2 - MiniMax H3 Reference to Video",
        133: "PASS 2 - SamplerCustomAdvanced",
    }

    for node in data["nodes"]:
        if node["id"] in expected:
            assert node.get("title") == expected[node["id"]]
        else:
            assert "title" not in node
