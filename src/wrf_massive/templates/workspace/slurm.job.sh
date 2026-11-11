#!/bin/bash

# SLURM job script for `wrf-massive submit <sim_dir>` (set as `job_file` in env.yaml). Nodes, tasks, CPUs, memory
# and walltime are set by `submit` from each stage's `Resources` in pipeline.py; add site-specific options
# (partition, account, ...) here.
##SBATCH --partition=<partition>

STAGE=$1
SIM_DIR=$2

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm.job.sh <stage> <sim_dir>"
  exit 1
fi

# Load modules needed by WPS/WRF on your cluster, e.g.
# module load foss/2024a

# Run (unbuffered). `submit` passes the env file it was started with via $WRF_MASSIVE_ENV_FILE.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run wrf-massive run --stages="$STAGE" "$SIM_DIR"
