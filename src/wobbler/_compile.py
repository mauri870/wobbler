"""torch.compile of the hot functions on every device, the CPU included. A function is compiled on its first call (dynamic shapes, so that the sizes of a run do not recompile it); where torch cannot compile it, one warning is given and the eager function runs. The number of compiled variants torch keeps is raised the way transitgpu raises it. The environment variable WOBBLER_COMPILE=0, or set_compile(False), runs every function eagerly. TORCHINDUCTOR_CACHE_DIR names the directory of the compiled kernels (set it on disk, not in a RAM-backed /tmp)."""
import os
import warnings

import torch

RECOMPILE_LIMIT = 64
_ENABLED = os.environ.get("WOBBLER_COMPILE", "1") != "0"
_COMPILED = {}


def set_compile(enabled):
    """Turn the compilation of the hot functions on or off for the rest of the process."""
    global _ENABLED
    _ENABLED = bool(enabled)


def compile_enabled():
    return _ENABLED


def _raise_recompile_limit():
    from torch._dynamo import config

    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    if getattr(config, name) < RECOMPILE_LIMIT:
        setattr(config, name, RECOMPILE_LIMIT)


def hot(function):
    """The decorator of a hot function: calls it compiled when compilation is on and available, eagerly otherwise (and always with the same arguments and results)."""

    def run(*args):
        if _ENABLED and _COMPILED.get(function) is not False:
            try:
                _raise_recompile_limit()
                if function not in _COMPILED:
                    _COMPILED[function] = torch.compile(function, dynamic=True)
                return _COMPILED[function](*args)
            except torch.OutOfMemoryError:
                raise
            except Exception as error:  # no compiler, or a backend that cannot build the kernel
                warnings.warn(f"torch.compile of {function.__name__} failed ({type(error).__name__}: {str(error)[:200]}); running it eagerly", stacklevel=2)
                _COMPILED[function] = False
        return function(*args)

    run.__name__ = function.__name__
    run.eager = function
    return run
