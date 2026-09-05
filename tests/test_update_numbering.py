from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modifications_contains_updates_1_through_9():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")

    for number in range(1, 10):
        assert f"Update {number}" in text




def test_readme_stays_recent_changes_only_and_technical_report_has_dated_chronology():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "TECHNICAL_ARCHITECTURE.md").read_text(encoding="utf-8")
    expected = {
        1: "2026-08-10",
        2: "2026-08-12",
        3: "2026-08-14",
        4: "2026-08-14",
        5: "2026-08-15",
        6: "2026-08-17",
        7: "2026-08-18",
        8: "2026-08-30",
    }
    assert "## Update timeline" not in readme
    assert "complete dated update history" in readme
    assert "### Dated update chronology" in architecture
    for number, date in expected.items():
        assert f"Update {number} — **{date}" in architecture


def test_readme_update9_is_compact_and_links_detailed_guidance():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme.split()) < 800
    for item in ("Update 9", "24 fps", "39 + 51*k", "5 + 17*k", "1/4096",
                 "%date:yyyy-MM-dd%", "KEYFRAMES_AND_INSERTS.md",
                 "WORKFLOW_PREREQUISITES.md", "MODIFICATIONS.md"):
        assert item in readme

def test_modifications_numbered_update_headings_are_dated():
    modifications = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    expected = {
        1: "2026-08-10",
        2: "2026-08-12",
        3: "2026-08-14",
        4: "2026-08-14",
        5: "2026-08-15",
        6: "2026-08-17",
        7: "2026-08-18",
        8: "2026-08-30",
    }
    for number, date in expected.items():
        heading = next(
            line for line in modifications.splitlines()
            if line.startswith(f"## Update {number} —")
        )
        assert date in heading


def test_update_history_is_consolidated():
    assert (ROOT / "MODIFICATIONS.md").exists()
    assert (ROOT / "TECHNICAL_ARCHITECTURE.md").exists()

    for name in (
        "H3_MASKED_AV_BRIDGE.md",
        "NATIVE_CORE_15439.md",
        "UPDATE_3_2026-08-14.md",
        "UPDATE_4_2026-08-14.md",
        "UPDATE_5_2026-08-15.md",
    ):
        assert not (ROOT / name).exists()


def test_update_4_and_5_history_is_preserved_in_modifications():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")

    assert "Update 4" in text
    assert "2026-08-14" in text
    assert "Update 5" in text
    assert "2026-08-15" in text


def test_update_6_direct_latent_history_is_preserved():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    assert "Update 6" in text
    assert "2026-08-17" in text
    assert "checkpoint-free direct-latent" in text
    assert "direct VHS streaming" in text or "direct single-pass" in text
    assert "timebase" in text


def test_update_7_credits_pr3_and_reithan():
    modifications = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for text in (modifications, readme):
        assert "Update 7" in text
        assert "PR #3" in text
        assert "Reithan" in text


def test_update_8_is_distinct_from_pr3_update_7():
    modifications = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "TECHNICAL_ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "## Update 7 — Arbitrary-position latent inserts and keyframes" in modifications
    assert "## Update 8 — released 2026-08-30" in modifications
    assert "Update 9" in readme
    assert "## Update 8 — released 2026-08-30" in modifications
    assert "fractional H3 V2V denoise" in readme
    assert "current work in progress" not in readme
    assert "started 2026-08-30" not in readme
    assert "is in progress" not in readme
    assert "work in progress" not in modifications
    assert "(work in progress)" not in architecture


def test_update_8_contains_dynamic_derope_seam_without_renumbering_update_7():
    modifications = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    architecture = (ROOT / "TECHNICAL_ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "Step 1q: dynamic de-rope seam continuation" in modifications
    assert "target_start" in modifications
    assert "H3 Fan Recovered Context" in modifications
    assert "Update 7's arbitrary `insert_frame`" in modifications
    assert "Interior Motion Context placement for de-rope seams" in architecture


def test_current_docs_do_not_reintroduce_obsolete_v2v_or_hardcoded_bridge_wording():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workflows = (ROOT / "example_workflows" / "README.md").read_text(encoding="utf-8")
    modifications = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    v2v = workflows.split("## NEW - V2V Latent Motion Transfer", 1)[1].split("\n---", 1)[0]
    assert "custom-node folder" not in v2v.lower()
    assert "historical" not in v2v.lower()
    for text in (readme, workflows, modifications):
        assert not ("independent" in text.lower() and "hard-coded seconds" in text.lower())
