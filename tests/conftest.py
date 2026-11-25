"""The wobbler tests run the package eagerly (no torch.compile) unless a test turns compilation on, so that the fast suite does not wait for the compiler, and on the CPU unless a test names a device (the default device asks torch for a GPU, which the cpu phase of a merge hides and a local run may not)."""
import pytest

import wobbler
from wobbler import _tensor


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    wobbler.set_compile(False)
    monkeypatch.setattr(_tensor, "default_device", lambda: _tensor.torch.device("cpu"))
    yield
    wobbler.set_compile(False)
