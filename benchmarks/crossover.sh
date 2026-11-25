#!/bin/bash
# Three runs of benchmarks/crossover.py on one device, each as a fresh process on a quiet machine: over 60 s at least 97 percent idle and no process above 5 percent of a core (it checks up to 15 times).
#   bash benchmarks/crossover.sh PYTHON DEVICE DIRECTORY     (DEVICE: cpu or cuda; the cache of the compiled code is DIRECTORY/cache-DEVICE, kept between the runs)
# The idle share, the busiest process, the load average, the threads in uninterruptible sleep and the free memory go to DIRECTORY/crossover-session.jsonl before each run. A run whose file exists is skipped on a restart. Stop it by touching DIRECTORY/STOP.
set -u
python=${1:?python}
device=${2:?device}
directory=${3:?directory}
mkdir -p "$directory"
cd "$(dirname "$0")/.."
for n in 1 2 3; do
  name="crossover-$device-$n"
  [ -e "$directory/STOP" ] && { echo "stopped before $name"; exit 0; }
  [ -e "$directory/$name.json" ] && continue
  # quiet machine: over 60 s at least 97 percent idle (vmstat) and no process above 5 percent of a core, checked up to 15 times
  quiet=0
  for _ in $(seq 15); do
    idle=$(vmstat 60 2 | tail -1 | awk '{print $15}')
    busiest=$(ps -eo pcpu,comm --sort=-pcpu | sed -n 2p)
    top_pcpu=$(echo "$busiest" | awk '{print $1}')
    if awk -v i="$idle" -v c="$top_pcpu" 'BEGIN { exit !(i >= 97 && c <= 5.0) }'; then quiet=1; break; fi
  done
  [ "$quiet" = 1 ] || { echo "the machine was not quiet for 15 checks of 60 s: idle $idle percent, busiest $busiest" >&2; exit 1; }
  load=$(cut -d' ' -f1 /proc/loadavg)
  dstate=$(ps -eLo state,comm | awk '$1 == "D" {print $2}' | sort | uniq -c | awk '{printf "%s%s", sep, $2; sep=" "}')
  free=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "{\"run\": \"$name\", \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"idle_percent_over_60_s\": $idle, \"busiest_process\": \"$busiest\", \"load_1min\": $load, \"threads_in_uninterruptible_sleep\": \"$dstate\", \"memory_available_kib\": $free}" >> "$directory/crossover-session.jsonl"
  TORCHINDUCTOR_CACHE_DIR="$directory/cache-$device/inductor" TRITON_CACHE_DIR="$directory/cache-$device/triton" "$python" benchmarks/crossover.py "$device" "$directory/$name.json" > "$directory/$name.txt" 2>&1
  echo "$name: exit $?"
done
