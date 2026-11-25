"""wobbler.scan: the scan of one star (times, scan angles and along-scan parallax factors) read from a scanning-law table, and its windows."""
import numpy as np
import pytest

from wobbler import scan

pytestmark = pytest.mark.torch_cpu

HEADER = "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB], scanAngle[rad], parallaxFactorAlongScan, other\n"


def table(rows):
    return HEADER + "".join(f"{jd}, {psi}, {plx}, x\n" for jd, psi, plx in rows)


def test_a_table_is_read_with_its_three_columns_and_the_spaces_after_the_commas(tmp_path):
    path = tmp_path / "scan.csv"
    path.write_text(table([(2457000.5, 0.1, 0.3), (2457010.5, 0.2, -0.4)]))
    s = scan.Scan.from_gost(path)
    assert s.jd.tolist() == [2457000.5, 2457010.5] and s.psi.tolist() == [0.1, 0.2] and s.plx_factor.tolist() == [0.3, -0.4]


def test_the_epochs_are_sorted_by_time(tmp_path):
    path = tmp_path / "scan.csv"
    path.write_text(table([(2457010.5, 0.2, -0.4), (2457000.5, 0.1, 0.3)]))
    assert scan.Scan.from_gost(path).jd.tolist() == [2457000.5, 2457010.5]


def test_a_window_selects_the_epochs_inside_it_and_none_outside(tmp_path):
    path = tmp_path / "scan.csv"
    path.write_text(table([(2457000.5, 0.1, 0.3), (2457500.5, 0.2, -0.4), (2458000.5, 0.3, 0.1)]))
    s = scan.Scan.from_gost(path)
    assert s.window(2457000.0, 2457900.0).tolist() == [True, True, False]
    assert not s.window(2459000.0, 2459100.0).any()


def test_a_table_without_the_columns_is_an_error_that_names_them(tmp_path):
    path = tmp_path / "scan.csv"
    path.write_text("a, b\n1, 2\n")
    with pytest.raises(ValueError, match="ObservationTimeAtBarycentre"):
        scan.Scan.from_gost(path)


def test_the_time_in_years_is_relative_to_the_reference_epoch():
    s = scan.Scan(jd=np.array([2457389.0, 2457389.0 + 365.25]), psi=np.zeros(2), plx_factor=np.zeros(2))
    assert s.t_yr().tolist() == [0.0, 1.0]


def test_a_scan_of_no_epoch_and_a_scan_of_one_epoch_are_scans():
    none = scan.Scan(jd=np.array([]), psi=np.array([]), plx_factor=np.array([]))
    assert len(none) == 0 and none.window(2457000.0, 2458000.0).tolist() == [] and none.t_yr().tolist() == []
    one = scan.Scan(jd=2457389.0 + 365.25, psi=0.3, plx_factor=-0.2)
    assert len(one) == 1 and one.t_yr().tolist() == [1.0] and one.window(2457000.0, 2458000.0).tolist() == [True]


def test_an_unknown_window_is_refused_before_gaiamock_is_asked():
    with pytest.raises(ValueError, match="window"):
        scan.Scan.from_position(262.0, -0.5, window="dr9")


def test_without_gaiamock_from_position_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "gaiamock", None)
    monkeypatch.delitem(__import__("sys").modules, "gaiamock.gaiamock", raising=False)
    with pytest.raises(ImportError, match="pip install gaiamock"):
        scan.Scan.from_position(262.0, -0.5)


def scanning_law_or_skip():
    gaiamock = pytest.importorskip("gaiamock.gaiamock")
    try:
        gaiamock.get_gost_one_position(262.17120727595824, -0.5810919626714084, data_release="dr3", binned=True)
    except (FileNotFoundError, OSError):
        pytest.skip("gaiamock's scanning-law data is not downloaded (gaiamock-download; GAIAMOCK_DATA_DIR)")
    return gaiamock


def test_the_scan_at_the_position_of_gaia_bh1_is_the_50_epochs_of_the_dr3_window_of_gaiamocks_scanning_law():
    gaiamock = scanning_law_or_skip()
    s = scan.Scan.from_position(262.17120727595824, -0.5810919626714084, window="DR3")
    table = gaiamock.get_gost_one_position(262.17120727595824, -0.5810919626714084, data_release="dr3", binned=True)
    jd = np.sort(np.asarray(gaiamock.fetch_table_element("ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]", table), float))
    assert len(s) == 50
    assert s.jd.tolist() == jd.tolist() and bool(torch_all_sorted(s.jd))
    assert 2456891.5 < float(s.jd[0]) and float(s.jd[-1]) < 2457902.0


def torch_all_sorted(values):
    return bool((values[1:] >= values[:-1]).all())


def test_the_dr2_window_is_a_part_of_the_dr3_window_at_the_same_position():
    scanning_law_or_skip()
    dr3 = scan.Scan.from_position(262.17120727595824, -0.5810919626714084, window="DR3")
    dr2 = scan.Scan.from_position(262.17120727595824, -0.5810919626714084, window="DR2")
    assert 0 < len(dr2) < len(dr3) and float(dr2.jd[-1]) < 2457531.5
