"""The normalisation of the unit weight error that makes Gaia's RUWE: RUWE = UWE / u0(G, nu_eff), with u0 the table of Gaia DR3 (Lindegren 2023, RUWE_DR3.zip of the Gaia (E)DR3 auxiliary data, table_u0_g_c_p5.txt, for five-parameter solutions), a function of the G magnitude and of the effective wavenumber nu_eff_used_in_astrometry. The table is package data (tables/u0_dr3_p5.csv.gz, with its source and SHA-256 in tables/u0_dr3_p5.json); u0 interpolates it linearly inside its grid (G 4 to 21, nu_eff 1.0 to 2.0 per micron) and takes the nearest tabulated value outside it, as the ReadMe of the archive says."""
import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ._tensor import as_tensor, pick_device

DATA = Path(__file__).parent / "tables"


def _load(name):
    with gzip.open(DATA / name, "rt") as handle:
        rows = np.loadtxt(handle, delimiter=",", skiprows=1)
    g, c = np.unique(rows[:, 0]), np.unique(rows[:, 1])
    return g, c, rows[:, 2].reshape(len(g), len(c))


@lru_cache(maxsize=1)
def _table():
    return _load("u0_dr3_p5.csv.gz")


@lru_cache(maxsize=1)
def _table_dr2():
    return _load("u0_dr2_p5.csv.gz")


_ON_DEVICE = {}


def _table_on(device, table=_table):
    key = (str(device), table.__name__)
    if key not in _ON_DEVICE:
        _ON_DEVICE[key] = tuple(torch.as_tensor(x, device=device) for x in table())
    return _ON_DEVICE[key]


def u0(G, nu_eff, device=None):
    """u0(G, nu_eff) of Gaia DR3 for five-parameter solutions: G and nu_eff are numbers, NumPy arrays or tensors (broadcast against each other); the result is a float64 tensor of the broadcast shape."""
    return _interpolate(_table, G, nu_eff, device)


def u0_dr2(G, bp_rp, device=None):
    """u0(G, C) of Gaia DR2 (DR2_RUWE_V1.zip, table_u0_g_col.txt; tables/u0_dr2.json has its source and hash), C the BP-RP colour: RUWE = UWE / u0_dr2. Bilinear inside the grid (G 3.60 to 21.00, C -1.0 to 10.0) and the nearest value outside it, as u0 does for DR3. G and bp_rp are numbers, NumPy arrays or tensors, broadcast against each other; the result is a float64 tensor."""
    return _interpolate(_table_dr2, G, bp_rp, device)


def _interpolate(table, G, nu_eff, device):
    device = pick_device(G, nu_eff, device=device)
    g, c, grid = _table_on(device, table)
    G, nu = torch.broadcast_tensors(as_tensor(G, device), as_tensor(nu_eff, device))
    G, nu = torch.clamp(G, g[0], g[-1]), torch.clamp(nu, c[0], c[-1])
    i = torch.clamp(torch.searchsorted(g, G.contiguous(), right=True) - 1, 0, len(g) - 2)
    j = torch.clamp(torch.searchsorted(c, nu.contiguous(), right=True) - 1, 0, len(c) - 2)
    fg, fc = (G - g[i]) / (g[i + 1] - g[i]), (nu - c[j]) / (c[j + 1] - c[j])
    return (1 - fg) * (1 - fc) * grid[i, j] + fg * (1 - fc) * grid[i + 1, j] + (1 - fg) * fc * grid[i, j + 1] + fg * fc * grid[i + 1, j + 1]
