"""The frozen NumPy implementation (oracle_numpy.py) still returns what it returned when it was frozen: the shape and the sum of absolute values of each array of oracle_cases.build() against golden/oracle_digest.json. A function test of a recorded summary; NumPy only."""
import json
from pathlib import Path

import numpy as np

import oracle_cases

DIGEST = json.loads((Path(__file__).parent / "golden" / "oracle_digest.json").read_text())


def test_the_oracle_returns_the_recorded_arrays():
    built = oracle_cases.build()
    assert set(built) == set(DIGEST)
    for key, recorded in DIGEST.items():
        values = np.asarray(built[key], dtype=float)
        assert list(values.shape) == recorded["shape"], key
        assert float(np.sum(np.abs(values[np.isfinite(values)]))) == np.float64(recorded["abs_sum"]) or np.isclose(np.sum(np.abs(values[np.isfinite(values)])), recorded["abs_sum"], rtol=1e-9), key
