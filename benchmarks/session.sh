#!/bin/bash
# Three runs of benchmarks/bench.py on one device, each as a fresh process with an empty cache directory and then as a second process with the filled one.
#   bash benchmarks/session.sh PYTHON DEVICE DIRECTORY     (DEVICE: cpu or cuda)
# PYTHON is the interpreter of an environment with wobbler, torch and gaiamock. A run starts when the one-minute load average is under 1 (it waits up to 15 minutes), and the load average and the free memory
# are written before each run to DIRECTORY/session.jsonl. A run whose file exists is skipped on a restart. Stop it by touching DIRECTORY/STOP.
set -u
python=${1:?python}
device=${2:?device}
directory=${3:?directory}
mkdir -p "$directory"
cd "$(dirname "$0")/.."

wait_quiet() {
  for _ in $(seq 90); do
    load=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$load" 'BEGIN { exit !(l < 1.0) }' && return 0
    sleep 10
  done
  echo "the load average stayed at $load for 15 minutes" >&2
  return 1
}

record() {  # record NAME
  "$python" - "$directory" "$1" <<'PY'
import json, os, sys, time
directory, name = sys.argv[1:]
free = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable"))
with open(os.path.join(directory, "session.jsonl"), "a") as handle:
    handle.write(json.dumps({"run": name, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_1min": os.getloadavg()[0], "memory_available_kib": free}) + "\n")
PY
}

run() {  # run NAME CACHE_DIRECTORY
  name=$1
  cache=$2
  [ -e "$directory/STOP" ] && { echo "stopped before $name"; exit 0; }
  [ -e "$directory/$name.txt" ] && return
  wait_quiet || exit 1
  record "$name"
  TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" "$python" benchmarks/bench.py "$device" > "$directory/$name.txt" 2>&1
  echo "$name: exit $?"
}

for n in 1 2 3; do
  cache="$directory/cache-$n"
  rm -rf "$cache"
  mkdir -p "$cache"
  run "$device-$n-empty-cache" "$cache"
  run "$device-$n-filled-cache" "$cache"
done
