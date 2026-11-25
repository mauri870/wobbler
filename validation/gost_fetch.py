"""The scans of many positions from ESA's Gaia Observation Forecast Tool (GOST, https://gaia.esac.esa.int/gost/), fetched as its web form does and cached on disk.

usage: python validation/gost_fetch.py POSITIONS.csv OUTDIR [--from 2014-08-01T00:00:00] [--to 2017-05-29T00:00:00] [--batch 1000] [--pause 20]
POSITIONS.csv has one position a line, name,ra_degrees,dec_degrees (the format that GOST takes for an upload; the name has no comma). The positions go in batches of at most --batch (the default, 1,000, is the size measured to take about 110 s with the default pause; a batch of 5,000 was submitted once and its report was not listed on the result page at the first read; GOST's own limit is 10,000), one batch at a time, each as the form sends it: a GET of the page for its session cookie, a multipart POST to GostServlet (inputmode file, the CSV, from, to), the result page of the session, and the CSV export of the report. OUTDIR gets batch_NNNN.csv for each batch and batch_NNNN.json with its positions' names, the number of rows of each and the time of the fetch; a batch whose file exists is not fetched again. Scan.from_gost(path, target=name) reads one position of a file.
Pace: --pause seconds between requests and between batches, one request at a time, the default client name of urllib. A reply that is an error, a captcha or a status other than COMPLETED stops the run with GostError and nothing is retried; a report that the result page does not list is read for again, up to POLLS reads, and then the last page is saved and the run stops.
GOST is a forecast of the commanded scanning law: its page says that the predicted observation time may deviate by up to 6 hours from the actual one and that operational gaps and the gaps between CCDs are not modelled. Gaia data are distributed under CC BY-NC 3.0 IGO (https://www.cosmos.esa.int/web/gaia-users/license); ESA states no terms for GOST's results, and they are read here as covered by that licence (argued, not stated by ESA); keep them out of git and name GOST and the Gaia credit (https://www.cosmos.esa.int/web/gaia-users/credits) where a result uses them."""
import argparse
import http.cookiejar
import json
import re
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://gaia.esac.esa.int/gost/"
MAXIMUM_BATCH = 10_000
POLLS = 20  # result pages read for a report that is not complete yet; after them the run stops


class GostError(RuntimeError):
    """GOST answered with an error, a captcha or an unexpected status."""


class NotListed(GostError):
    """The result page does not list the report (yet)."""


def parse_submitted(body):
    """The report id of the reply to a submission: 'Submitted with id N'. Any other reply is a GostError carrying its text."""
    found = re.search(r"Submitted with id (\d+)", body)
    if found:
        return found.group(1)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    raise GostError(f"GOST did not accept the submission: {text[:300]}")


def parse_result(page, report_id):
    """{'status', 'message', 'csv'} of the report with id `report_id` on the result page: the status and message cells of its row and the link of its CSV export, relative to BASE."""
    link = re.search(rf'href="(export\.jsp\?id=[^"&]*/{re.escape(str(report_id))}&format=csv)"', page)
    if not link:
        raise NotListed(f"the result page does not list the report {report_id}")
    start, end = page.rfind("<tr", 0, link.start()), page.find("</tr>", link.end())
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td>(.*?)</td>", page[start:end], flags=re.S)]
    return {"status": cells[1], "message": cells[2], "csv": link.group(1)}


def batches(positions, size):
    """The positions in lists of at most `size` (GOST takes 10,000 at most), in order."""
    if size > MAXIMUM_BATCH or size < 1:
        raise ValueError(f"a batch holds 1 to {MAXIMUM_BATCH} positions, not {size}")
    return [positions[i:i + size] for i in range(0, len(positions), size)]


def positions_csv(positions):
    """The text that GOST takes for an upload: name,ra,dec in degrees, one position a line."""
    for name, _, _ in positions:
        if "," in str(name):
            raise ValueError(f"the name {name!r} has a comma, which GOST's CSV cannot hold")
    return "".join(f"{name},{ra!r},{dec!r}\n" for name, ra, dec in positions)


def targets_of(export):
    """{target: rows} of the CSV export of a report."""
    counts = {}
    for line in export.splitlines()[1:]:
        if line.strip():
            name = line.split(",")[0]
            counts[name] = counts.get(name, 0) + 1
    return counts


def check_export(export, names):
    """The names of the batch that have no row in the export (a position that no field of view crossed); a target in the export that is not in the batch is a GostError."""
    found = targets_of(export)
    strangers = sorted(set(found) - set(names))
    if strangers:
        raise GostError(f"the export has rows of targets that were not submitted: {strangers[:5]}")
    return [name for name in names if name not in found]


def multipart(fields, files):
    """(body, content type) of a multipart/form-data upload, the encoding of the form."""
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    for name, (filename, content) in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n'.encode() + content.encode() + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def fetch_batch(positions, start, end, pause, opener, save_to=None):
    """The CSV export of one batch, by the flow of the web form. A report that the result page does not list yet is waited for within the POLLS reads of the page; when it is still not listed after them the last page is saved in `save_to` (result_page_<id>.html) and the run stops with GostError."""
    def get(path):
        with opener.open(BASE + path, timeout=300) as reply:
            return reply.read().decode("utf-8", errors="replace")

    get("")
    time.sleep(pause)
    body, kind = multipart({"inputmode": "file", "srcname": "", "serviceCode": "1", "srcra": "0.0", "srcdec": "0.0", "from": start, "to": end}, {"csvfilename": ("targets.csv", positions_csv(positions))})
    request = urllib.request.Request(BASE + "GostServlet", data=body, headers={"Content-Type": kind})
    with opener.open(request, timeout=600) as reply:
        report_id = parse_submitted(reply.read().decode("utf-8", errors="replace"))
    page = ""
    for _ in range(POLLS):
        time.sleep(pause)
        page = get("result.jsp")
        try:
            report = parse_result(page, report_id)
        except NotListed:
            continue
        if report["status"] == "COMPLETED":
            time.sleep(pause)
            return get(report["csv"])
        if report["status"] not in ("RUNNING", "SUBMITTED", "QUEUED", "PENDING"):
            raise GostError(f"the report {report_id} has the status {report['status']!r}: {report['message']}")
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        (Path(save_to) / f"result_page_{report_id}.html").write_text(page)
    raise GostError(f"the report {report_id} was not complete after {POLLS} reads of the result page" + (f"; the last page is saved in {save_to}" if save_to is not None else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("positions"), parser.add_argument("out")
    parser.add_argument("--from", dest="start", default="2014-08-01T00:00:00"), parser.add_argument("--to", default="2017-05-29T00:00:00")
    parser.add_argument("--batch", type=int, default=1000), parser.add_argument("--pause", type=float, default=20.0)
    arguments = parser.parse_args()
    positions = [(n, float(a), float(d)) for n, a, d in (line.split(",") for line in Path(arguments.positions).read_text().splitlines() if line.strip())]
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    for number, batch in enumerate(batches(positions, arguments.batch)):
        target = out / f"batch_{number:04d}.csv"
        if target.exists():
            print(f"batch {number}: cached", flush=True)
            continue
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        export = fetch_batch(batch, arguments.start, arguments.to, arguments.pause, opener, save_to=out)
        missing = check_export(export, [name for name, _, _ in batch])
        target.with_suffix(".tmp").write_text(export)
        target.with_suffix(".tmp").rename(target)
        (out / f"batch_{number:04d}.json").write_text(json.dumps({"names": [name for name, _, _ in batch], "rows": targets_of(export), "without_rows": missing, "from": arguments.start, "to": arguments.to,
                                                                 "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": BASE}, indent=1) + "\n")
        print(f"batch {number}: {len(batch)} positions, {sum(targets_of(export).values())} rows, {len(missing)} without rows", flush=True)
        time.sleep(arguments.pause)


if __name__ == "__main__":
    main()
