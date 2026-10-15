from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"


def test_example_workflow_catalog_is_tight_and_current():
    names = sorted(p.name for p in WF.glob("*.json"))
    assert names == sorted([
        "NEW - 2MP De-Rope Continuation - Working Example.json",
        "NEW - AV Extension.json",
        "NEW - Music Video (Compact Loop).json",
        "NEW - Music Video.json",
        "NEW - V2V Latent Motion Transfer (with upscale and de-rope).json",
        "OLD - Hybrid Extension.json",
        "OLD - Motion Context - Advanced.json",
        "OLD - Motion Context - Simple.json",
        "UTILITY - AV Bridge.json",
        "UTILITY - Custom Keyframes.json",
    ])
    assert not any("Live Latent" in name for name in names)
    assert not any("Latent Masking" in name for name in names)


def test_current_workflows_highlighted_as_new_are_exact():
    names = sorted(p.name for p in WF.glob("NEW - *.json"))
    assert names == [
        "NEW - 2MP De-Rope Continuation - Working Example.json",
        "NEW - AV Extension.json",
        "NEW - Music Video (Compact Loop).json",
        "NEW - Music Video.json",
        "NEW - V2V Latent Motion Transfer (with upscale and de-rope).json",
    ]
