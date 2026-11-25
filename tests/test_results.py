"""wobbler.results.to_table: one row per case, the columns of COLUMNS, a missing nu_eff as NaN, a missing window as an error, and the DataFrame form."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from wobbler import Orbit, Scan, simulate_fit  # noqa: E402
from wobbler.results import COLUMNS, to_table  # noqa: E402

pytestmark = pytest.mark.torch_cpu


def fit(nu_eff=1.5):
    rng = np.random.default_rng(12)
    scan = Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, 50)), psi=rng.uniform(0, np.pi, 50), plx_factor=rng.uniform(-1, 1, 50), device="cpu")
    return simulate_fit(scan, Orbit(period_days=150.0, ecc=0.1, t_peri_days=0.0, omega=0.3), a1_au=0.5, parallax_mas=np.array([1.0, 3.0]), G=12.0, n_orient=7, seed=5, nu_eff=nu_eff, device="cpu")


def test_the_table_has_one_row_per_case_and_the_documented_columns_in_order():
    table = to_table(fit(), [1.0, 3.0])
    assert tuple(table) == COLUMNS
    assert all(len(column) == 14 for column in table.values())
    assert list(table["parallax_true_mas"][:7]) == [1.0] * 7 and list(table["parallax_true_mas"][7:]) == [3.0] * 7
    assert list(table["orientation_index"][:7]) == list(range(7))
    assert np.isfinite(table["ruwe"]).all() and table["ok"].dtype == bool


def test_without_nu_eff_the_ruwe_column_is_nan_and_the_uwe_column_is_not():
    table = to_table(fit(nu_eff=None), 1.0 * np.array([1.0, 3.0]))
    assert np.isnan(table["ruwe"]).all() and np.isfinite(table["uwe"]).all()


def test_an_unknown_window_names_the_windows_the_fit_has():
    with pytest.raises(KeyError, match="DR3"):
        to_table(fit(), [1.0, 3.0], window="DR9")


def test_the_dataframe_form_has_the_same_columns_or_names_the_extra():
    try:
        import pandas
    except ImportError:
        with pytest.raises(ImportError, match=r"wobbler\[pandas\]"):
            to_table(fit(), [1.0, 3.0], frame=True)
        return
    frame = to_table(fit(), [1.0, 3.0], frame=True)
    assert isinstance(frame, pandas.DataFrame) and list(frame.columns) == list(COLUMNS)
