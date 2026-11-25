"""Scans of a star from a Gaia source id or from epoch astrometry, which the core Scan does not read.

scan_from_source_id looks up the position of a Gaia DR3 source and returns the scan of gaiamock's scanning law there (the `gaia` and `gaiamock` extras; it asks the Gaia archive for one row).
scan_from_epoch_astrometry builds a Scan from the per-transit table of Gaia's epoch astrometry. It is provisional: it reads the column names of the draft data model of Gaia DR4 (listed in DRAFT_COLUMNS) and has not been run on the release's own table.
"""
import numpy as np

ARCHIVE_URL = "https://gea.esac.esa.int/tap-server/tap"
TIME_ORIGIN_JD = 2455197.5  # J2010.0, the origin of the time columns of the draft epoch astrometry (TCB)
NS_PER_DAY = 86400e9
DRAFT_COLUMNS = {"time_tcb_ns": "obs_time_tcb", "bary_corr_ns": "obs_time_bary_corr", "scan_angle_deg": "scan_pos_angle", "parallax_factor_al": "parallax_factor_al", "used": "used_by_agis_al"}


def position_of_source_id(source_id, release="gaiadr3"):
    """(ra_deg, dec_deg) of a Gaia source id at the epoch of the catalogue, by one query of the Gaia archive (network; needs astroquery)."""
    try:
        from astroquery.gaia import Gaia
    except ImportError as error:
        raise ImportError("position_of_source_id needs astroquery: install wobbler[gaia] (Installation in the README)") from error
    rows = Gaia.launch_job(f"select ra, dec from {release}.gaia_source where source_id = {int(source_id)}").get_results()
    if len(rows) != 1:
        raise LookupError(f"the Gaia archive has {len(rows)} rows of {release}.gaia_source for source_id {int(source_id)}")
    return float(rows[0]["ra"]), float(rows[0]["dec"])


def scan_from_source_id(source_id, window="DR3", device=None):
    """The Scan at the position of a Gaia DR3 source id: position_of_source_id, then Scan.from_position (which needs gaiamock)."""
    from .scan import Scan

    ra, dec = position_of_source_id(source_id)
    return Scan.from_position(ra, dec, window=window, device=device)


def scan_from_epoch_astrometry(table, columns=None, device=None):
    """The Scan of one source from its epoch astrometry (PROVISIONAL: the column names are those of the draft Gaia DR4 data model).

    table: a mapping from column name to array (a dict, a NumPy structured array or a pandas DataFrame), for one source. With the draft names: obs_time_tcb and obs_time_bary_corr (nanoseconds since J2010.0), scan_pos_angle (degrees) and parallax_factor_al, one row per transit; arrays of shape (transits, CCDs) are reduced to one value per transit (the mean time and parallax factor, the first finite scan angle), and rows with used_by_agis_al false are left out when that column is present.
    columns: a dictionary that replaces some of DRAFT_COLUMNS (the keys time_tcb_ns, bary_corr_ns, scan_angle_deg, parallax_factor_al, used) when the release names them differently.
    """
    from .scan import Scan

    names = {**DRAFT_COLUMNS, **(columns or {})}

    def get(key, required=True):
        if names[key] not in table and not (hasattr(table, "dtype") and getattr(table.dtype, "names", None) and names[key] in table.dtype.names):
            if required:
                raise KeyError(f"the table has no column {names[key]!r} ({key}); pass columns={{{key!r}: ...}} to name it")
            return None
        return np.asarray(table[names[key]], dtype=float)

    def per_transit(values, first=False):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values
        if first:
            return np.array([row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan for row in values])
        return np.nanmean(values, axis=1)

    tcb, bary = get("time_tcb_ns"), get("bary_corr_ns")
    if tcb.ndim == 2 and bary.ndim == 1:
        bary = bary[:, None]
    jd = TIME_ORIGIN_JD + per_transit((tcb + bary) / NS_PER_DAY)
    psi = np.radians(per_transit(get("scan_angle_deg"), first=True))
    factor = per_transit(get("parallax_factor_al"))
    used = get("used", required=False)
    keep = np.isfinite(jd) & np.isfinite(psi) & np.isfinite(factor)
    if used is not None:
        keep &= (per_transit(used) > 0.5)
    if not keep.any():
        raise ValueError("no transit of the table has a time, a scan angle and a parallax factor")
    order = np.argsort(jd[keep], kind="stable")
    return Scan(jd=jd[keep][order], psi=psi[keep][order], plx_factor=factor[keep][order], device=device)
