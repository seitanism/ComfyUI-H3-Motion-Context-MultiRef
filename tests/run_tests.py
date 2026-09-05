"""Run each CPU/static test module in a fresh pytest process.

ComfyUI mocks cannot leak between modules, and pytest fixtures retain their
normal behavior. No ComfyUI installation or H3 weights are needed.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess
import sys

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def run_all() -> int:
    passed = 0
    failures = []
    env = os.environ.copy()
    # The outer aggregate pytest test may be active; child runs are independent.
    env.pop("PYTEST_CURRENT_TEST", None)
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == "test_repo_suite.py":
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(path),
             "-o", "python_files=test_*.py"],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        matches = re.findall(r"(\d+) passed", proc.stdout)
        count = int(matches[-1]) if matches else 0
        if proc.returncode or not count:
            failures.append(path.name)
            print(proc.stdout, flush=True)
        else:
            passed += count
            print(f"PASS {path.name}: {count} checks", flush=True)
    if failures:
        raise RuntimeError("Failed test modules: " + ", ".join(failures))
    print(f"PASS: {passed} repo CPU/static checks", flush=True)
    return passed


if __name__ == "__main__":
    run_all()
