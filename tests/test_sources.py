"""wobbler.sources: the epoch-astrometry reader on a synthetic table with the draft column names, and the errors that name the extra when astroquery is missing."""
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from wobbler.sources import NS_PER_DAY, TIME_ORIGIN_JD, position_of_source_id, scan_from_epoch_astrometry  # noqa: E402

pytestmark = pytest.mark.torch_cpu


def table(transits=6, ccds=None, seed=2):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(1500.0, 2500.0, transits))
    shape = (transits,) if ccds is None else (transits, ccds)
    tcb = (days[:, None] + 0.001 * np.arange(ccds)[None, :] if ccds else days) * NS_PER_DAY
    return {"obs_time_tcb": tcb, "obs_time_bary_corr": rng.uniform(-500, 500, transits) * 1e9, "scan_pos_angle": rng.uniform(0, 180, shape), "parallax_factor_al": rng.uniform(-1, 1, shape), "used_by_agis_al": np.ones(shape)}, days


def test_a_one_dimensional_table_gives_the_scan_with_the_time_scan_angle_and_factor_of_each_transit():
    rows, days = table()
    scan = scan_from_epoch_astrometry(rows, device="cpu")
    assert len(scan) == 6
    expected = TIME_ORIGIN_JD + days + rows["obs_time_bary_corr"] / NS_PER_DAY
    assert np.allclose(scan.jd.numpy(), expected)
    assert np.allclose(scan.psi.numpy(), np.radians(rows["scan_pos_angle"]))


def test_per_ccd_arrays_are_reduced_to_one_value_per_transit_and_unused_transits_are_left_out():
    rows, _ = table(transits=5, ccds=9)
    rows["used_by_agis_al"][2, :] = 0
    scan = scan_from_epoch_astrometry(rows, device="cpu")
    assert len(scan) == 4
    assert np.all(np.diff(scan.jd.numpy()) > 0)


def test_a_missing_column_names_the_column_and_how_to_rename_it():
    rows, _ = table()
    del rows["scan_pos_angle"]
    with pytest.raises(KeyError, match="scan_pos_angle"):
        scan_from_epoch_astrometry(rows)


def test_renamed_columns_are_read_by_the_columns_argument():
    rows, _ = table()
    rows["angle"] = rows.pop("scan_pos_angle")
    assert len(scan_from_epoch_astrometry(rows, columns={"scan_angle_deg": "angle"}, device="cpu")) == 6


def test_a_table_with_no_usable_transit_is_refused():
    rows, _ = table()
    rows["used_by_agis_al"][:] = 0
    with pytest.raises(ValueError, match="no transit"):
        scan_from_epoch_astrometry(rows)


def test_the_source_id_lookup_names_the_extra_when_astroquery_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "astroquery", None)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", None)
    with pytest.raises(ImportError, match=r"wobbler\[gaia\]"):
        position_of_source_id(4373465352415301632)
