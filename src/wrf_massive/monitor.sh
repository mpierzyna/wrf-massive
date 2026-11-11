#!/bin/bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Log memory and disk usage of a running stage every INTERVAL seconds, until terminated (SIGTERM/SIGINT).
# Started by `wrf-massive run` when $WRF_MASSIVE_MONITOR is set. Uses standard Linux tools only.
#
# Usage: monitor.sh INTERVAL WORK_DIR OUT_PREFIX
#   <OUT_PREFIX>_mem.csv:  timestamp,elapsed_s,used_mib,process_mib,peak_mib,limit_mib,source
#   <OUT_PREFIX>_disk.csv: timestamp,elapsed_s,size_kib   (du of WORK_DIR, following it if it is a symlink)
#
# Memory comes from the SLURM job's cgroup (all processes of the job on this node), so other jobs on a shared node
# are not counted. `used` includes the file cache (e.g. of written output), which the kernel frees before killing a
# job; `process` (anonymous memory) is what counts for running out of memory. Outside a SLURM job, or without a job
# cgroup, the whole node is reported (source=node). The test-only variables _WRF_MASSIVE_CGROUP_ROOT and
# _WRF_MASSIVE_PROC_CGROUP replace /sys/fs/cgroup and /proc/self/cgroup.

INTERVAL=$1
WORK_DIR=$2
OUT_PREFIX=$3
CGROUP_ROOT=${_WRF_MASSIVE_CGROUP_ROOT:-/sys/fs/cgroup}
PROC_CGROUP=${_WRF_MASSIVE_PROC_CGROUP:-/proc/self/cgroup}
MEM_CSV="${OUT_PREFIX}_mem.csv"
DISK_CSV="${OUT_PREFIX}_disk.csv"
START=$(date +%s)
SLEEP_PID=

mib() { [ -n "$1" ] && echo $(($1 / 1048576)); }
stat_value() { awk -v key="$1" '$1 == key {print $2; exit}' "$2" 2>/dev/null; }
read_file() { [ -r "$1" ] && cat "$1"; }

# The job-level cgroup (job_<id>) contains the batch script and all job steps (e.g. MPI ranks) on this node.
CG_VERSION=
CG_DIR=
if [ -n "${SLURM_JOB_ID:-}" ]; then
  job="job_${SLURM_JOB_ID}"
  if [ -f "$CGROUP_ROOT/cgroup.controllers" ]; then
    path=$(awk -F: '$1 == "0" {print $3}' "$PROC_CGROUP")
    base="$CGROUP_ROOT"
    CG_VERSION=2
  else
    path=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' "$PROC_CGROUP")
    base="$CGROUP_ROOT/memory"
    CG_VERSION=1
  fi
  case "$path/" in  # cut the path after job_<id>, e.g. /.../job_42/step_batch/user/task_0 -> /.../job_42
    */"$job"/*) path="$path/" && CG_DIR="$base${path%%/"$job"/*}/$job" ;;
  esac
  [ -d "$CG_DIR" ] || CG_VERSION=
fi

sample_mem() {
  local used process peak limit source
  if [ "$CG_VERSION" = 2 ]; then
    used=$(read_file "$CG_DIR/memory.current")
    process=$(stat_value anon "$CG_DIR/memory.stat")
    peak=$(read_file "$CG_DIR/memory.peak")
    limit=$(read_file "$CG_DIR/memory.max")
    [ "$limit" = max ] && limit=
    source=cgroup2
  elif [ "$CG_VERSION" = 1 ]; then
    used=$(read_file "$CG_DIR/memory.usage_in_bytes")
    process=$(stat_value total_rss "$CG_DIR/memory.stat")
    peak=$(read_file "$CG_DIR/memory.max_usage_in_bytes")
    limit=$(read_file "$CG_DIR/memory.limit_in_bytes")
    [ "${#limit}" -ge 19 ] && limit=  # "unlimited" is (close to) the 64-bit maximum
    source=cgroup1
  else
    local total available anon
    total=$(stat_value MemTotal: /proc/meminfo)
    available=$(stat_value MemAvailable: /proc/meminfo)
    anon=$(stat_value AnonPages: /proc/meminfo)
    used=$(((total - available) * 1024))
    process=$((anon * 1024))
    limit=$((total * 1024))
    peak=
    source=node
  fi
  echo "$1,$(mib "$used"),$(mib "$process"),$(mib "$peak"),$(mib "$limit"),$source" >>"$MEM_CSV"
}

sample_disk() {
  local size
  size=$(du -s -k -D "$WORK_DIR" 2>/dev/null | awk '{print $1}')
  echo "$1,$size" >>"$DISK_CSV"
}

sample() {
  local now
  now="$(date -Iseconds),$(($(date +%s) - START))"
  sample_mem "$now"
  sample_disk "$now"
}

stop() {
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null
  sample
  exit 0
}
trap stop TERM INT

echo "timestamp,elapsed_s,used_mib,process_mib,peak_mib,limit_mib,source" >"$MEM_CSV"
echo "timestamp,elapsed_s,size_kib" >"$DISK_CSV"
while true; do
  sample
  sleep "$INTERVAL" &
  SLEEP_PID=$!
  wait "$SLEEP_PID"
done
