"""wobbler.light (NumPy) against the torch functions of wobbler.orbit and wobbler.mass on random inputs, and the import of wobbler.light where torch is not installed."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wobbler import light

torch = pytest.importorskip("torch")
from wobbler import mass, orbit  # noqa: E402

RNG = np.random.default_rng(20260122)  # seed of the inputs
N = 500


def close(a, b, tolerance=1e-10):
    np.testing.assert_allclose(np.asarray(a, dtype=float), b.cpu().numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("ecc_high", [0.8, 0.97])
def test_kepler_matches_the_torch_solver_for_low_and_high_eccentricity(ecc_high):
    mean, ecc = RNG.uniform(-30, 30, N), RNG.uniform(0, ecc_high, N)
    close(light.kepler(mean, ecc), orbit.kepler(mean, ecc, device="cpu"))
    assert np.max(np.abs(light.kepler(mean, ecc) - ecc * np.sin(light.kepler(mean, ecc)) - mean)) < 1e-10


def test_kepler_refuses_an_eccentricity_outside_zero_to_one_and_accepts_an_empty_array():
    with pytest.raises(ValueError, match="eccentricity"):
        light.kepler(1.0, 1.0)
    assert light.kepler(np.array([]), np.array([])).shape == (0,)


def test_the_orbit_functions_match_the_torch_ones():
    a, node, omega, incl = RNG.uniform(0.1, 5, N), RNG.uniform(0, 6.28, N), RNG.uniform(0, 6.28, N), RNG.uniform(0.1, 1.5, N)
    for ours, theirs in zip(light.thiele_innes(a, node, omega, incl), orbit.thiele_innes(a, node, omega, incl, device="cpu")):
        close(ours, theirs)
    A, B, F, G = light.thiele_innes(a, node, omega, incl)
    elements, reference = light.campbell(A, B, F, G), orbit.campbell(A, B, F, G, device="cpu")
    for name in ("a", "node", "omega", "inclination"):
        close(elements[name], reference[name])
    period, ecc, t_peri, k = RNG.uniform(5, 500, N), RNG.uniform(0, 0.9, N), RNG.uniform(-100, 100, N), RNG.uniform(1, 50, N)
    t = RNG.uniform(-300, 300, N)
    close(light.radial_velocity(t, period, ecc, t_peri, omega, k), orbit.radial_velocity(t, period, ecc, t_peri, omega, k, device="cpu"))
    for ours, theirs in zip(light.photocentre(t / 365.25, period, ecc, t_peri, a, incl, node, omega), orbit.photocentre(t / 365.25, period, ecc, t_peri, a, incl, node, omega, device="cpu")):
        close(ours, theirs)
    close(light.true_anomaly(t, ecc), orbit.true_anomaly(t, ecc, device="cpu"))


def test_the_campbell_elements_invert_the_thiele_innes_constants():
    a, node, omega, incl = 2.0, 1.0, 2.0, 0.7
    elements = light.campbell(*light.thiele_innes(a, node, omega, incl))
    assert np.allclose([elements["a"], elements["inclination"]], [a, incl])
    assert np.allclose(light.thiele_innes(elements["a"], elements["node"], elements["omega"], elements["inclination"]), light.thiele_innes(a, node, omega, incl))


def test_the_mass_functions_match_the_torch_ones_and_a_zero_mass_function_has_zero_mass():
    period, k, ecc = RNG.uniform(5, 3000, N), RNG.uniform(1, 100, N), RNG.uniform(0, 0.9, N)
    close(light.mass_function(period, k, ecc), mass.mass_function(period, k, ecc, device="cpu"))
    f, m1, incl = light.mass_function(period, k, ecc), RNG.uniform(0.5, 3, N), RNG.uniform(0.2, 1.5, N)
    close(light.min_companion_mass(f, m1), mass.min_companion_mass(f, m1, device="cpu"))
    close(light.companion_mass(f, m1, incl), mass.companion_mass(f, m1, incl, device="cpu"))
    assert light.min_companion_mass(0.0, 1.0) == 0.0
    flux, q = RNG.uniform(0, 3, N), RNG.uniform(0.2, 3, N)
    close(light.photocentre_ratio(flux, q), mass.photocentre_ratio(flux, q, device="cpu"))
    ratio = RNG.uniform(-0.5, 1.0, N)
    close(light.flux_ratio_from_ratio(ratio, q), mass.flux_ratio_from_ratio(ratio, q, device="cpu"))
    close(light.reflex_semi_major_axis(k, period, ecc), mass.reflex_semi_major_axis(k, period, ecc, device="cpu"))
    close(light.share_inclined_below(incl), mass.share_inclined_below(incl, device="cpu"))
    assert light.photocentre_ratio(0.0, 1.0) == 1.0, "a dark companion: the photocentre is the primary"


def test_the_constants_equal_those_of_the_torch_modules():
    assert (light.G_SI, light.SOLAR_MASS_KG, light.AU_KM, light.DANBY_UPDATES, light.DAYS_PER_YEAR) == (mass.G_SI, mass.SOLAR_MASS_KG, mass.AU_KM, orbit.DANBY_UPDATES, orbit.DAYS_PER_YEAR)


def test_light_imports_and_runs_where_torch_cannot_be_imported(tmp_path):
    """A subprocess whose `import torch` fails: wobbler.light works, and a torch name raises the ImportError that names the extra."""
    (tmp_path / "torch.py").write_text("raise ModuleNotFoundError(\"No module named 'torch'\", name='torch')\n")
    code = "import wobbler.light as l; print(float(l.kepler(1.0, 0.1)));\ntry:\n    import wobbler; wobbler.simulate_fit\nexcept ImportError as error:\n    print(error)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env={"PYTHONPATH": f"{tmp_path}:{Path(__file__).resolve().parents[1] / 'src'}", "PATH": ""})
    assert out.returncode == 0, out.stderr
    assert "wobbler[torch]" in out.stdout, out.stdout
