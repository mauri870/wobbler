"""Every script of examples/ runs, as the README says, on the CPU and with no data."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted(p for p in (Path(__file__).resolve().parents[1] / "examples").glob("*.py") if not p.name.startswith("_"))


def test_the_examples_were_found():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_the_example_runs_and_prints(path):
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_the_example_runs_with_no_user_warning(path):
    out = subprocess.run([sys.executable, "-W", "error::UserWarning", str(path)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
