"""The factor by which Gaia DR3's unit weight error of a five-parameter star exceeds the one this package simulates for it, as a function of the G magnitude. Gaia's UWE of an ordinary star is above one at bright G (up to about 1.9 times, near G 11.3) because the real scatter of the epochs and the formal errors of the fit do not agree; the simulated single star has a UWE near one.

This factor is the one of a star with no orbit signal. simulate_fit applies it to the noise of the simulated unit weight error and the orbit factor (orbit_factor, below) to the orbit's part by default (`uwe_factor=1` gives the raw simulation, as gaiamock does, and `uwe_factor=wobbler.uwe_factor` the product of this factor and the raw value). The table (tables/uwe_factor_dr3.json, with its sample, query conditions and SHA-256) holds the factor in bands of 0.5 mag from G 4.5 to 19 and of 0.1 mag from G 10.5 to 12: the median UWE sqrt(chi2_al / (n_good_obs_al - 5)) of ordinary DR3 stars of the band over the UWE that this package simulates for a single star there. The factor is the linear interpolation between the band centres and holds the end values outside them.

Measured range: on 2,000 DR3 stars with orbital solutions, the product of this factor and the simulated UWE is within 5 percent of the archive's (median over the stars of a G bin) in every bin except G 11 to 12, where it is about 12 percent high; on 60,940 such stars the product is within 5 percent in 5 of 9 bins of G and 19 percent high at its worst (G 11.25 to 11.5). The orbit factor below removes it. The table is replaced when the factor is measured again."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE = Path(__file__).parent / "tables" / "uwe_factor_dr3.json"
TABLE_DR2 = Path(__file__).parent / "tables" / "uwe_factor_dr2.json"


@lru_cache(maxsize=2)
def _nodes(table=TABLE):
    bands = json.loads(table.read_text())["bands"]
    return np.array([b.get("G_node", 0.5 * (b["G"][0] + b["G"][1])) for b in bands]), np.array([b["f"] for b in bands])


def uwe_factor(G):
    """The factor of Gaia DR3 for G a number or an array: a float64 NumPy array of the shape of G (a 0-d array for a number)."""
    centres, values = _nodes()
    return np.interp(np.asarray(G, dtype=float), centres, values)


def uwe_factor_dr2(G):
    """The factor of Gaia DR2 for G a number or an array, as uwe_factor is the factor of DR3: the median UWE of ordinary DR2 stars of the band (tables/uwe_factor_dr2.json, with its sample, simulation and hash) over the UWE that this package simulates for a single star in the DR2 window; the linear interpolation between the node positions (the median G of the rows of a node, G_node, or the band centre where the table has none), the end values outside them. A float64 NumPy array of the shape of G."""
    centres, values = _nodes(TABLE_DR2)
    return np.interp(np.asarray(G, dtype=float), centres, values)


ORBIT_TABLE = Path(__file__).parent / "tables" / "uwe_orbit_factor_dr3.json"
ORBIT_EDGES = tuple(0.25 * k + 9.0 for k in range(17))  # the bands of G of the orbit table: 9 to 13 in steps of 0.25 mag


@lru_cache(maxsize=1)
def _orbit_bands():
    bands = json.loads(ORBIT_TABLE.read_text())["bands"]
    return np.array([b["G"][0] for b in bands] + [bands[-1]["G"][1]]), np.array([b["B"] for b in bands])


def _lookup(g, values):
    edges, _ = _orbit_bands()
    index = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(values) - 1)
    return np.where(g >= edges[-1], uwe_factor(g), np.asarray(values, dtype=float)[index])


def orbit_factor(G):
    """The factor B(G) on the orbit's part of the unit weight error of Gaia DR3 for G a number or an array (a float64 NumPy array of the shape of G). The table (tables/uwe_orbit_factor_dr3.json, with its sample, fit and check) holds B in bands of G of 0.25 mag from 9 to 13; B is the value of the band that holds G, the first band below G 9, and the single-star factor f(G) from G 13 on, where the sample ends."""
    return _lookup(np.asarray(G, dtype=float), _orbit_bands()[1])


def calibrated_uwe(G, uwe_simulated, bands=None):
    """The unit weight error of Gaia DR3 predicted for a star of magnitude G whose simulated unit weight error is `uwe_simulated`: sqrt(f(G)^2 + B(G)^2 (w^2 - 1)), with w the simulated value, f the single-star factor (uwe_factor) and B the orbit factor (orbit_factor, or the array `bands` of one value per band of the table if given). A star whose simulated value is 1 gets f(G), a strong signal gets B(G) times w, and from G 13 on, where B is f, the form is f times w for every w. The sum under the root is kept at 0 or more."""
    g, w = np.broadcast_arrays(np.asarray(G, dtype=float), np.asarray(uwe_simulated, dtype=float))
    b = orbit_factor(g) if bands is None else _lookup(g, bands)
    return np.sqrt(np.maximum(uwe_factor(g) ** 2 + b**2 * (w**2 - 1.0), 0.0))


def fit_orbit_factor(G, uwe_simulated, uwe_archive, minimum_uwe=1.5):
    """The orbit factor of each band of the table from stars with an orbit: the square root of the median, over the stars of the band with a simulated unit weight error of `minimum_uwe` or more, of max(0, (uwe_archive^2 - f^2) / (uwe_simulated^2 - 1)). Returns one value per band of the table. This is the fit that made tables/uwe_orbit_factor_dr3.json."""
    g, w, archive = (np.asarray(x, dtype=float) for x in (G, uwe_simulated, uwe_archive))
    f = uwe_factor(g)
    out = []
    for low, high in zip(ORBIT_EDGES[:-1], ORBIT_EDGES[1:]):
        m = (g >= low) & (g < high) & (w >= minimum_uwe)
        out.append(float(np.sqrt(np.median(np.maximum(0.0, (archive[m] ** 2 - f[m] ** 2) / (w[m] ** 2 - 1.0))))))
    return np.array(out)


ORBIT_TABLE_DR2 = Path(__file__).parent / "tables" / "uwe_orbit_factor_dr2.json"


@lru_cache(maxsize=1)
def _orbit_bands_dr2():
    bands = json.loads(ORBIT_TABLE_DR2.read_text())["bands"]
    return np.array([b["G"][0] for b in bands] + [bands[-1]["G"][1]]), np.array([b["B"] for b in bands])


def orbit_factor_dr2(G):
    """The orbit factor B of Gaia DR2 for G a number or an array (a float64 NumPy array of the shape of G): the value of the band of tables/uwe_orbit_factor_dr2.json that holds G (0.25 mag bands from G 9 to 15), the first band below G 9, and the single-star factor uwe_factor_dr2(G) from G 15 on."""
    g = np.asarray(G, dtype=float)
    edges, values = _orbit_bands_dr2()
    index = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(values) - 1)
    return np.where(g >= edges[-1], uwe_factor_dr2(g), values[index])


def calibrated_uwe_dr2(G, uwe_simulated):
    """The unit weight error of Gaia DR2 predicted for a star of magnitude G whose simulated unit weight error in the DR2 window is `uwe_simulated`: sqrt(f(G)^2 + B(G)^2 (w^2 - 1)) with f = uwe_factor_dr2 and B = orbit_factor_dr2, the sum under the root kept at 0 or more. Divide by u0_dr2(G, bp_rp) for the RUWE."""
    g, w = np.broadcast_arrays(np.asarray(G, dtype=float), np.asarray(uwe_simulated, dtype=float))
    return np.sqrt(np.maximum(uwe_factor_dr2(g) ** 2 + orbit_factor_dr2(g) ** 2 * (w**2 - 1.0), 0.0))


UNBINNED_TABLE = Path(__file__).parent / "tables" / "uwe_factor_unbinned_dr3.json"
UNBINNED_EDGES = tuple(0.25 * k + 9.0 for k in range(29))  # the bands of the unbinned table: G 9 to 16 in steps of 0.25 mag


@lru_cache(maxsize=1)
def _unbinned_nodes():
    bands = json.loads(UNBINNED_TABLE.read_text())["bands"]
    return np.array([b.get("G_node", 0.5 * (b["G"][0] + b["G"][1])) for b in bands]), np.array([b["f"] for b in bands])


def unbinned_factor(G):
    """The factor by which Gaia DR3's RUWE of a single star exceeds the RUWE of the unbinned mode (simulate_fit with binned=False and uwe_factor=1), for G a number or an array (a float64 NumPy array of the shape of G). The table (tables/uwe_factor_unbinned_dr3.json, with its sample, fit and held-out check) holds the median ratio in bands of 0.25 mag from G 9 to 16; the factor is the linear interpolation between the band centres. Outside the centres, below G 9.125 and above G 15.875, it holds the end values: the sample ends at G 9 and G 16, so the factor there is not measured."""
    centres, values = _unbinned_nodes()
    return np.interp(np.asarray(G, dtype=float), centres, values)


def fit_unbinned_factor(G, ruwe_archive, ruwe_unbinned):
    """The factor of each band of the unbinned table from single stars: the median archive RUWE over the median RUWE of the unbinned mode (uwe_factor=1) of the stars of the band. Returns one value per band. This is the fit that made tables/uwe_factor_unbinned_dr3.json."""
    g, archive, simulated = (np.asarray(x, dtype=float) for x in (G, ruwe_archive, ruwe_unbinned))
    out = []
    for low, high in zip(UNBINNED_EDGES[:-1], UNBINNED_EDGES[1:]):
        m = (g >= low) & (g < high)
        out.append(float(np.median(archive[m]) / np.median(simulated[m])))
    return np.array(out)
