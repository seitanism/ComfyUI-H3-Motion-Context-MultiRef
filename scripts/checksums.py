"""Generate or verify the complete release SHA256SUMS manifest."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def manifest_text():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST
                   and not any(part in IGNORED for part in p.relative_to(ROOT).parts)
                   and p.suffix != ".pyc")
    return "".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  ./" +
                   p.relative_to(ROOT).as_posix() + "\n" for p in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on missing, extra, or mismatched entries")
    args = parser.parse_args()
    text = manifest_text()
    if args.check:
        if not MANIFEST.exists() or MANIFEST.read_text() != text:
            raise SystemExit("SHA256SUMS.txt is stale. Run python scripts/checksums.py after final edits.")
        print("All release checksums verified.")
    else:
        MANIFEST.write_text(text)
        print(f"Wrote {len(text.splitlines())} checksums.")


if __name__ == "__main__":
    main()
