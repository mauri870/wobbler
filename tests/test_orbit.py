"""wobbler.orbit: Kepler's equation on arrays, the Thiele-Innes constants and their inverse, the radial velocity and the photocentre orbit. Test functions: the cases are closed forms and round trips, with an independent per-case solver where the code is vectorised."""
import numpy as np
import pytest

from wobbler import orbit

pytestmark = pytest.mark.torch_cpu


def bisect_kepler(mean_anomaly, ecc):
    """E - e sin E = M by bisection on [0, 2 pi] after wrapping M, one case at a time (an oracle that shares no code with the solver)."""
    wrapped = np.mod(mean_anomaly, 2 * np.pi)
    low, high = 0.0, 2 * np.pi
    for _ in range(200):
        mid = 0.5 * (low + high)
        if mid - ecc * np.sin(mid) < wrapped:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high) + (mean_anomaly - wrapped)


def test_kepler_satisfies_the_equation_to_1e_12_over_the_whole_range_of_eccentricity():
    rng = np.random.default_rng(1)
    ecc = np.concatenate([[0.0, 0.5, 0.99, 0.999], rng.uniform(0, 0.99, 200)])
    mean = rng.uniform(-20, 20, (len(ecc), 30))
    solved = orbit.kepler(mean, ecc[:, None]).numpy()
    assert np.max(np.abs(solved - ecc[:, None] * np.sin(solved) - mean)) < 1e-11


def test_kepler_agrees_with_an_independent_bisection_at_high_eccentricity():
    rng = np.random.default_rng(2)
    for ecc in (0.0, 0.3, 0.9, 0.99):
        mean = rng.uniform(-10, 10, 25)
        expected = np.array([bisect_kepler(m, ecc) for m in mean])
        assert np.max(np.abs(orbit.kepler(mean, ecc).numpy() - expected)) < 1e-10, ecc


def test_kepler_edge_cases_are_exact():
    assert orbit.kepler(np.array([0.0]), 0.5)[0] == pytest.approx(0.0, abs=1e-14)
    assert orbit.kepler(np.array([np.pi]), 0.9)[0] == pytest.approx(np.pi, abs=1e-12)
    assert orbit.kepler(np.array([1.234]), 0.0)[0] == pytest.approx(1.234, abs=1e-14), "circular: E equals M"
    with pytest.raises(ValueError, match="eccentricity"):
        orbit.kepler(np.array([1.0]), 1.0)


def test_a_face_on_circular_orbit_is_a_circle_of_the_semi_major_axis():
    t = np.linspace(0, 365.25, 7) / 365.25
    x, y = orbit.photocentre(t, period_days=365.25, ecc=0.0, t_peri_days=0.0, a0_mas=3.0, inclination=0.0, node=0.4, omega=1.1)
    assert np.allclose(np.hypot(x, y), 3.0, atol=1e-12)


def test_an_edge_on_orbit_moves_along_a_line_through_the_centre():
    t = np.linspace(0, 1, 41)
    x, y = orbit.photocentre(t, period_days=300.0, ecc=0.3, t_peri_days=10.0, a0_mas=2.0, inclination=np.pi / 2, node=0.7, omega=0.9)
    assert np.linalg.matrix_rank(np.column_stack([x, y]), tol=1e-9) == 1
    angle = np.arctan2(x, y)  # the line is at the node's position angle, measured from north through east
    assert np.allclose(np.mod(angle, np.pi), np.mod(0.7, np.pi), atol=1e-9)


def test_the_thiele_innes_constants_invert_to_the_elements():
    rng = np.random.default_rng(3)
    a0 = rng.uniform(0.5, 5, 50)
    node = rng.uniform(0, 2 * np.pi, 50)
    omega = rng.uniform(0, 2 * np.pi, 50)
    inclination = rng.uniform(0.1, np.pi - 0.1, 50)
    constants = orbit.thiele_innes(a0, node, omega, inclination)
    back = orbit.campbell(*constants)
    again = orbit.thiele_innes(back["a"], back["node"], back["omega"], back["inclination"])
    assert all(np.allclose(p, q, atol=1e-10) for p, q in zip(constants, again))
    assert np.allclose(back["a"], a0)


def test_the_photocentre_matches_the_orbital_plane_rotation_for_a_case_by_hand():
    """At periastron of an eccentric orbit the offset is (X, Y) = (a (1 - e), 0) in the orbital plane; with omega = 0 and node = 0 and edge-on the north offset is a (1 - e) and the east offset 0."""
    x, y = orbit.photocentre(np.array([0.0]), period_days=365.25, ecc=0.4, t_peri_days=0.0, a0_mas=2.0, inclination=np.pi / 2, node=0.0, omega=0.0)
    assert x[0] == pytest.approx(0.0, abs=1e-12) and y[0] == pytest.approx(2.0 * (1 - 0.4), rel=1e-12)


def test_the_radial_velocity_of_a_circular_orbit_is_a_sinusoid_of_the_semi_amplitude():
    t = np.linspace(0, 100, 200, endpoint=False)
    v = orbit.radial_velocity(t, period_days=50.0, ecc=0.0, t_peri_days=0.0, omega=0.0, k=12.0).numpy()
    assert np.max(v) == pytest.approx(12.0, rel=1e-6) and np.min(v) == pytest.approx(-12.0, rel=1e-6) and np.mean(v) == pytest.approx(0.0, abs=1e-6)


def test_the_batched_photocentre_equals_the_case_by_case_one():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(-1, 3, 40))
    cases = {"period_days": rng.uniform(100, 800, 9), "ecc": rng.uniform(0, 0.8, 9), "t_peri_days": rng.uniform(-100, 100, 9), "a0_mas": rng.uniform(0.5, 4, 9),
             "inclination": rng.uniform(0.2, 3.0, 9), "node": rng.uniform(0, 6.28, 9), "omega": rng.uniform(0, 6.28, 9)}
    bx, by = orbit.photocentre(t[None, :], **{k: v[:, None] for k, v in cases.items()})
    for i in range(9):
        x, y = orbit.photocentre(t, **{k: float(v[i]) for k, v in cases.items()})
        assert np.allclose(bx[i], x, atol=1e-12) and np.allclose(by[i], y, atol=1e-12)


def test_the_photocentre_by_hand_for_other_nodes_and_arguments_of_periastron():
    """Edge-on, from the Thiele-Innes constants: node 90 degrees and omega 0 put the periastron offset a (1 - e) to the east; node 0 and omega 90 degrees put the quarter-period offset of a circular orbit a to the south; a face-on orbit with node 0 and omega 0 starts north and goes east."""
    x, y = orbit.photocentre(np.array([0.0]), period_days=365.25, ecc=0.4, t_peri_days=0.0, a0_mas=2.0, inclination=np.pi / 2, node=np.pi / 2, omega=0.0)
    assert x[0] == pytest.approx(2.0 * (1 - 0.4), rel=1e-12) and y[0] == pytest.approx(0.0, abs=1e-12)
    x, y = orbit.photocentre(np.array([0.25]), period_days=365.25, ecc=0.0, t_peri_days=0.0, a0_mas=2.0, inclination=np.pi / 2, node=0.0, omega=np.pi / 2)
    assert x[0] == pytest.approx(0.0, abs=1e-12) and y[0] == pytest.approx(-2.0, rel=1e-12)
    x, y = orbit.photocentre(np.array([0.0, 0.25]), period_days=365.25, ecc=0.0, t_peri_days=0.0, a0_mas=2.0, inclination=0.0, node=0.0, omega=0.0)
    assert np.allclose([x[0], y[0], x[1], y[1]], [0.0, 2.0, 2.0, 0.0], atol=1e-12)


def test_an_orbit_carries_only_the_elements_of_its_shape_and_its_time_of_periastron():
    """Orientations are arguments of the forward model; an Orbit that held a node and an inclination would be ignored by it."""
    import dataclasses

    assert [f.name for f in dataclasses.fields(orbit.Orbit)] == ["period_days", "ecc", "t_peri_days", "omega"]
