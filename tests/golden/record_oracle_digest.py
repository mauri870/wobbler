"""Records tests/golden/oracle_digest.json: for each array that oracle_cases.build() returns, its shape and the sum of its absolute values (the finite ones), so that a test sees an edit of the frozen NumPy implementation.
usage: python tests/golden/record_oracle_digest.py   (from the repository root; numpy only)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oracle_cases  # noqa: E402

digest = {key: {"shape": list(np.shape(v)), "abs_sum": float(np.sum(np.abs(np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))])))} for key, v in oracle_cases.build().items()}
(Path(__file__).parent / "oracle_digest.json").write_text(json.dumps(digest, indent=1, sort_keys=True))
print(len(digest), "arrays")
