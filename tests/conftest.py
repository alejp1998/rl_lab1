"""Pytest config: make the lab problem modules importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for problem_dir in (ROOT / "problem1", ROOT / "problem2"):
    sys.path.insert(0, str(problem_dir))
