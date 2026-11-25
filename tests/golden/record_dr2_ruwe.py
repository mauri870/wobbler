"""Records tests/golden/dr2_ruwe_stars.csv.gz: 1,000 Gaia DR2 five-parameter stars with a colour, seeded by random_index, with the columns the DR2 RUWE needs and the archive's own ruwe. Needs the network.
usage: python tests/golden/record_dr2_ruwe.py
The query joins gaiadr2.gaia_source with gaiadr2.ruwe (https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_ruwe.html: source_id, ruwe) on source_id."""
import csv
import gzip
import io
import urllib.parse
import urllib.request
from pathlib import Path

ADQL = """SELECT TOP 1000 g.source_id, g.phot_g_mean_mag, g.bp_rp, g.astrometric_chi2_al, g.astrometric_n_good_obs_al, r.ruwe
FROM gaiadr2.gaia_source AS g JOIN gaiadr2.ruwe AS r USING (source_id)
WHERE g.astrometric_params_solved = 31 AND g.bp_rp IS NOT NULL AND g.astrometric_n_good_obs_al > 20 AND g.random_index < 300000"""
URL = "https://gea.esac.esa.int/tap-server/tap/sync"
OUT = Path(__file__).with_name("dr2_ruwe_stars.csv.gz")

data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": ADQL}).encode()
with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=300) as response:
    text = response.read().decode()
rows = list(csv.reader(io.StringIO(text)))
assert len(rows) == 1001, f"the query returned {len(rows) - 1} rows"
with gzip.open(OUT, "wt", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerows(rows)
print(f"{len(rows) - 1} stars written to {OUT}")
