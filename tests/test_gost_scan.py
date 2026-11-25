"""Scan.from_gost on a batch export of ESA's Gaia Observation Forecast Tool (GOST, https://gaia.esac.esa.int/gost/): the rows of one target of a batch, in a window, and the agreement of GOST at a sky node's own position with the node's table in gaiamock.

tests/golden/gost_batch_3_positions.csv is the export of one batch of four positions, 2014-08-01 to 2017-05-29 (GOST version 22.4.3, read on 2026-11-24), without the fourth. The target node0 is the position of the node of gaiamock's healpix_64_0.fits, and tests/golden/gaiamock_node_0_dr3.csv is that file's rows in the same window."""
from pathlib import Path

import numpy as np
import pytest

from wobbler import scan
from wobbler.gaiafit import WINDOWS

pytestmark = pytest.mark.torch_cpu

GOLDEN = Path(__file__).parent / "golden"
BATCH, NODE = GOLDEN / "gost_batch_3_positions.csv", GOLDEN / "gaiamock_node_0_dr3.csv"
TIME = "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"


def rows_of(target):
    lines = BATCH.read_text().splitlines()
    header = [name.strip() for name in lines[0].split(",")]
    return [dict(zip(header, line.split(","))) for line in lines[1:] if line.split(",")[0] == target]


@pytest.mark.parametrize("target, count", [("node0", 46), ("bh3", 47), ("welopoai_Aa", 64)])
def test_the_rows_of_one_target_of_a_batch_are_its_scan_sorted_by_time(target, count):
    s = scan.Scan.from_gost(BATCH, target=target)
    expected = sorted(float(r[TIME]) for r in rows_of(target))
    assert len(s) == count and s.jd.tolist() == expected, (len(s), count)


def test_a_batch_read_without_a_target_has_every_row_and_a_target_the_file_lacks_names_the_ones_it_has():
    assert len(scan.Scan.from_gost(BATCH)) == 46 + 47 + 64
    with pytest.raises(ValueError, match="bh1.*node0"):
        scan.Scan.from_gost(BATCH, target="bh1")


def test_the_window_keeps_the_rows_strictly_inside_it_and_a_window_that_is_not_one_is_refused():
    full = scan.Scan.from_gost(BATCH, target="bh3")
    low, high = WINDOWS["DR2"]
    inside = scan.Scan.from_gost(BATCH, target="bh3", window="DR2")
    assert inside.jd.tolist() == [t for t in full.jd.tolist() if low < t < high] and 0 < len(inside) < len(full)
    assert len(scan.Scan.from_gost(BATCH, target="bh3", window="DR3")) == sum(WINDOWS["DR3"][0] < t < WINDOWS["DR3"][1] for t in full.jd.tolist())
    with pytest.raises(ValueError, match="window"):
        scan.Scan.from_gost(BATCH, window="DR9")


def test_gost_at_the_position_of_a_node_gives_the_nodes_transits_within_the_difference_of_the_two_versions_of_gost():
    """Measured on 2026-11-24: 46 transits in both, the times differ by a median of 5.8 s and at most 10.7 s, the scan angles by at most 6.4e-6 rad and the along-scan parallax factors by at most 4.2e-6. The tolerances are about 1.5 times those."""
    exact, node = scan.Scan.from_gost(BATCH, target="node0"), scan.Scan.from_gost(NODE)
    assert len(exact) == len(node) == 46
    assert np.abs((exact.jd - node.jd).numpy()).max() < 1.9e-4 and np.abs((exact.psi - node.psi).numpy()).max() < 1e-5 and np.abs((exact.plx_factor - node.plx_factor).numpy()).max() < 7e-6, (
        np.abs((exact.jd - node.jd).numpy()).max(), np.abs((exact.psi - node.psi).numpy()).max(), np.abs((exact.plx_factor - node.plx_factor).numpy()).max())


def test_a_directory_of_batches_gives_every_target_of_every_file_and_each_equals_the_scan_read_from_its_file_alone(tmp_path):
    """The two paths agree (from_gost_batches reads a directory once; from_gost reads one file for one target)."""
    text = BATCH.read_text().splitlines()
    second = [text[0]] + ["other," + line.split(",", 1)[1] for line in text[1:] if line.startswith("bh3,")][:5]
    (tmp_path / "batch_0000.csv").write_text("\n".join(text) + "\n")
    (tmp_path / "batch_0001.csv").write_text("\n".join(second) + "\n")
    (tmp_path / "batch_0000.json").write_text("{}")
    scans = scan.Scan.from_gost_batches(tmp_path, window="DR3")
    assert sorted(scans) == ["bh3", "node0", "other", "welopoai_Aa"] and len(scans["other"]) == 5
    for target in ("bh3", "node0", "welopoai_Aa"):
        alone = scan.Scan.from_gost(BATCH, target=target, window="DR3")
        assert scans[target].jd.tolist() == alone.jd.tolist() and scans[target].psi.tolist() == alone.psi.tolist() and scans[target].plx_factor.tolist() == alone.plx_factor.tolist()
    with pytest.raises(ValueError, match="no batch_"):
        scan.Scan.from_gost_batches(tmp_path / "missing")
