"""The medians and ranges of the runs that benchmarks/session.sh wrote.

usage: python benchmarks/summarise.py DIRECTORY   (reads DIRECTORY/{cpu,cuda}-N-{empty,filled}-cache.txt, writes DIRECTORY/summary.json)
"""
import glob
import json
import re
import statistics
import sys
from pathlib import Path

directory = Path(sys.argv[1])
summary = {}
for device in ("cpu", "cuda"):
    rows = {"first_call_empty_cache": [], "first_call_filled_cache": [], "batches_of_1000_later_calls": [], "one_batch_of_14400": [], "gaiamock": []}
    for name in sorted(glob.glob(str(directory / f"{device}-*.txt"))):
        text = Path(name).read_text()
        rows["first_call_empty_cache" if "empty" in name else "first_call_filled_cache"].append(float(re.search(r"first call ([\d.]+) s", text).group(1)))
        rows["batches_of_1000_later_calls"].append(float(re.search(r"later calls ([\d.]+) s", text).group(1)))
        rows["one_batch_of_14400"].append(float(re.search(r"one batch of 14,400: ([\d.]+) s", text).group(1)))
        rows["gaiamock"].append(float(re.search(r"one case at a time: ([\d.]+) s", text).group(1)))
    summary[device] = {key: {"seconds": values, "median": statistics.median(values), "min": min(values), "max": max(values)} for key, values in rows.items()}
reference = [seconds for device in summary.values() for seconds in device["gaiamock"]["seconds"]]
summary["gaiamock_all_runs"] = {"seconds": reference, "median": statistics.median(reference), "min": min(reference), "max": max(reference)}
for device in ("cpu", "cuda"):
    summary[device]["speedup_over_gaiamock"] = {key: summary["gaiamock_all_runs"]["median"] / summary[device][key]["median"] for key in ("batches_of_1000_later_calls", "one_batch_of_14400")}
(directory / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({d: summary[d]["speedup_over_gaiamock"] for d in ("cpu", "cuda")}))
