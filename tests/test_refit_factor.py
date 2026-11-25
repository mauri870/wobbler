"""The split rule and the node construction of validation/dr2/refit_factor.py on synthetic stars with a known step in the unit weight error."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("refit_factor", Path(__file__).resolve().parents[1] / "validation" / "dr2" / "refit_factor.py")
refit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refit)


def band(step, seed=0, n=2500):
    rng = np.random.default_rng(seed)
    g = rng.uniform(13.0, 13.5, n)
    uwe = np.where(g < 13.2, 1.06, 0.78 if step else 1.06) * np.exp(rng.normal(0, 0.1, n))
    return g, uwe


def test_a_step_inside_the_band_splits_it_and_the_nodes_follow_the_step():
    g, uwe = band(step=True)
    nodes = refit.nodes_of_band(g, uwe, 13.0, 13.5, np.random.default_rng(1))
    assert len(nodes) == 5
    assert [round(n["G"][0], 2) for n in nodes] == [13.0, 13.1, 13.2, 13.3, 13.4]
    assert abs(nodes[0]["median_uwe"] - 1.06) < 0.02 and abs(nodes[4]["median_uwe"] - 0.78) < 0.02


def test_a_flat_band_is_not_split():
    g, uwe = band(step=False)
    nodes = refit.nodes_of_band(g, uwe, 13.0, 13.5, np.random.default_rng(1))
    assert len(nodes) == 1 and nodes[0]["G"] == [13.0, 13.5]


def test_a_band_with_a_sub_bin_of_fewer_than_fifty_rows_is_not_split_even_with_a_step():
    g, uwe = band(step=True, n=150)
    nodes = refit.nodes_of_band(g, uwe, 13.0, 13.5, np.random.default_rng(1))
    assert len(nodes) == 1


def test_an_empty_band_gives_no_error_and_a_single_node_with_no_rows():
    nodes = refit.nodes_of_band(np.array([]), np.array([]), 13.0, 13.5, np.random.default_rng(1))
    assert len(nodes) == 1 and nodes[0]["n_fit"] == 0


def test_a_node_sits_at_the_median_g_of_its_rows_not_at_its_band_centre():
    rng = np.random.default_rng(0)
    g = rng.uniform(13.0, 13.02, 2000)
    nodes = refit.nodes_of_band(g, np.full(2000, 1.0), 13.0, 13.5, np.random.default_rng(1))
    assert len(nodes) == 1 and abs(nodes[0]["G_node"] - 13.01) < 0.002


def test_a_step_at_the_band_edge_is_a_misfit_of_the_interpolation_and_splits_the_band():
    rng = np.random.default_rng(0)
    g = rng.uniform(13.0, 13.5, 3000)
    uwe = np.full(3000, 1.06) * np.exp(rng.normal(0, 0.1, 3000))
    subs = refit.sub_bins(g, uwe, 13.0, 13.5, np.random.default_rng(1))
    below, above = {"G_node": 12.75, "median_uwe": 1.06}, {"G_node": 13.75, "median_uwe": 0.78}
    assert not refit.pair_split(subs) and refit.misfit_split(subs, below, above)


def test_a_band_on_the_line_between_its_neighbours_is_not_a_misfit_and_a_missing_neighbour_is_no_test():
    rng = np.random.default_rng(0)
    g = rng.uniform(13.0, 13.5, 3000)
    uwe = np.interp(g, [12.0, 14.5], [1.0, 0.8]) * np.exp(rng.normal(0, 0.1, 3000))
    subs = refit.sub_bins(g, uwe, 13.0, 13.5, np.random.default_rng(1))
    assert not refit.misfit_split(subs, {"G_node": 12.0, "median_uwe": 1.0}, {"G_node": 14.5, "median_uwe": 0.8})
    assert not refit.misfit_split(subs, None, {"G_node": 14.5, "median_uwe": 0.8})


def test_rows_in_the_lowest_sliver_of_a_band_are_flagged_and_spread_rows_are_not():
    assert refit.sliver_sampled(np.linspace(13.5, 13.515, 500), 13.5, 14.0)
    assert not refit.sliver_sampled(np.linspace(13.5, 14.0, 500), 13.5, 14.0)
    assert refit.sliver_sampled(np.array([]), 13.5, 14.0)
