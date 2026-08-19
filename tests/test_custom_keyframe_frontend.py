from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_custom_keyframe_frontend_is_exported():
    init_text = (ROOT / "__init__.py").read_text()
    assert 'WEB_DIRECTORY = "./js"' in init_text
    assert '"WEB_DIRECTORY"' in init_text
    assert (ROOT / "js" / "h3_custom_keyframes.js").is_file()


def test_both_node_names_in_js():
    js_text = (ROOT / "js" / "h3_custom_keyframes.js").read_text()
    assert "MiniMaxH3CustomKeyframes" in js_text, \
        "JS must reference the original soft keyframes node name"
    assert "MiniMaxH3CustomKeyframesMasked" in js_text, \
        "JS must reference the new masked keyframes node name"
    # Must use a set/collection check (NODE_NAMES) rather than a single constant (NODE_NAME =).
    assert "const NODE_NAME " not in js_text, \
        "JS must not use a single NODE_NAME constant; use NODE_NAMES set instead"
    assert "NODE_NAMES" in js_text, \
        "JS must declare NODE_NAMES to cover both node types"
