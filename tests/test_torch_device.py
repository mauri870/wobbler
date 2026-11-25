"""The default device of wobbler on a machine with a GPU: the GPU when torch sees one, else the CPU. It asks torch for a GPU, so it runs in the gpu phase of a merge and not in the cpu phase."""
import pytest
import torch

import wobbler
from wobbler import _tensor

pytestmark = pytest.mark.gpu


def test_the_default_device_is_the_gpu_when_torch_sees_one_and_the_cpu_otherwise(monkeypatch):
    monkeypatch.undo()  # the conftest of the directory forces the CPU; this test needs the real function
    import importlib

    real = importlib.reload(_tensor)
    assert real.default_device().type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert wobbler.Scan([1.0], [0.0], [0.0]).jd.device.type in ("cuda", "cpu")
