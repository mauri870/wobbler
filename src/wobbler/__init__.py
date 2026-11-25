"""wobbler: a batched forward model of the orbital wobble of a star, on torch: Kepler's equation, Thiele-Innes and Campbell orbits, the photocentre orbit, the mass function, and Gaia's five-parameter astrometric fit (RUWE, excess noise, parallax) for many orbits at once, on the CPU or a GPU.

The names are loaded when first used, so that `import wobbler.light` (the orbit and mass functions in NumPy) works where torch is not installed; every other name needs torch (`pip install wobbler[torch]`).
"""
import importlib
from importlib.metadata import PackageNotFoundError, version

from .uwe_factor import calibrated_uwe, orbit_factor, unbinned_factor, uwe_factor  # needs only NumPy; bound here so that the function, not the submodule of the same name, is the attribute whatever is imported first

try:
    __version__ = version("wobbler")
except PackageNotFoundError:  # the source tree on PYTHONPATH, not installed
    __version__ = "0+unknown"

__all__ = ["__version__", "calibrated_uwe", "orbit_factor", "unbinned_factor", "epoch_astrometry", "joker", "light", "unbinned", "WINDOWS", "SimFit", "epoch_error", "fit_five_parameter", "observe", "simulate_fit", "companion_mass", "flux_ratio_from_ratio", "photocentre_ratio", "reflex_semi_major_axis", "mass_function", "min_companion_mass", "share_inclined_below", "Orbit", "campbell", "kepler", "photocentre", "radial_velocity", "thiele_innes", "true_anomaly", "Scan", "u0", "uwe_factor", "default_device", "set_compile", "compile_enabled"]

_MODULES = ("epoch_astrometry", "joker", "light", "unbinned")
_FROM = {
    "_compile": ("compile_enabled", "set_compile"),
    "_tensor": ("default_device",),
    "gaiafit": ("WINDOWS", "SimFit", "epoch_error", "fit_five_parameter", "observe", "simulate_fit"),
    "mass": ("companion_mass", "flux_ratio_from_ratio", "photocentre_ratio", "reflex_semi_major_axis", "mass_function", "min_companion_mass", "share_inclined_below"),
    "orbit": ("Orbit", "campbell", "kepler", "photocentre", "radial_velocity", "thiele_innes", "true_anomaly"),
    "ruwe": ("u0",),
    "scan": ("Scan",),
}
_HOME = {name: module for module, names in _FROM.items() for name in names}


def _load(module):
    try:
        return importlib.import_module(f".{module}", __name__)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ImportError(f"wobbler.{module.lstrip('_')} needs torch: install wobbler[torch] (Installation in the README)") from error
        raise


def __getattr__(name):
    if name in _MODULES:
        return _load(name)
    if name in _HOME:
        module = _load(_HOME[name])
        value = getattr(module, name)
        return value
    raise AttributeError(f"module 'wobbler' has no attribute {name!r}")
