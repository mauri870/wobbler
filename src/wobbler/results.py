"""Results of wobbler as plain arrays or a table.

to_table(fit) turns a SimFit of simulate_fit into a dictionary of NumPy arrays with one row per simulated case (a true parallax and an orientation of the orbit); with frame=True it is a pandas DataFrame
(the `pandas` extra). Nothing here needs torch: the fit's tensors are read through their `cpu().numpy()`.
"""
import numpy as np

COLUMNS = ("parallax_index", "orientation_index", "parallax_true_mas", "inclination_deg", "node_deg", "a0_mas", "uwe", "ruwe", "excess_noise_mas", "parallax_fit_mas", "n_epochs", "ok")


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def to_table(fit, parallax_mas, window=None, frame=False):
    """One row per case of `fit` (a SimFit) for one window: the columns are COLUMNS.

    parallax_mas: the true parallaxes that were given to simulate_fit (a number or an array), repeated on the rows. window: the name of the window, the first one of the fit (DR3 by default) if None.
    `ruwe` is NaN when simulate_fit was not given nu_eff. frame=True returns a pandas DataFrame with the same columns.
    """
    name = next(iter(fit.windows)) if window is None else window
    if name not in fit.windows:
        raise KeyError(f"the fit has no window {name!r}; it has {list(fit.windows)}")
    values = fit.windows[name]
    shape = _array(values["uwe"]).shape
    parallax = np.broadcast_to(np.atleast_1d(np.asarray(parallax_mas, dtype=float))[:, None], shape)
    index = np.indices(shape)
    ruwe = _array(values["ruwe"]) if "ruwe" in values else np.full(shape, np.nan)
    table = {
        "parallax_index": index[0].reshape(-1),
        "orientation_index": index[1].reshape(-1),
        "parallax_true_mas": parallax.reshape(-1),
        "inclination_deg": np.degrees(np.broadcast_to(_array(fit.inclination), shape)).reshape(-1),
        "node_deg": np.degrees(np.broadcast_to(_array(fit.node), shape)).reshape(-1),
        "a0_mas": np.broadcast_to(_array(fit.a0_mas), shape).reshape(-1),
        "uwe": _array(values["uwe"]).reshape(-1),
        "ruwe": ruwe.reshape(-1),
        "excess_noise_mas": _array(values["excess_noise"]).reshape(-1),
        "parallax_fit_mas": _array(values["parallax_mas"]).reshape(-1),
        "n_epochs": _array(values["n_epochs"]).reshape(-1),
        "ok": _array(values["ok"]).reshape(-1).astype(bool),
    }
    if not frame:
        return table
    try:
        import pandas
    except ImportError as error:
        raise ImportError("results.to_table(frame=True) needs pandas: install wobbler[pandas] (Installation in the README)") from error
    return pandas.DataFrame(table)
