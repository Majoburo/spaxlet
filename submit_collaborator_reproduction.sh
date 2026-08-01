#!/bin/bash
#SBATCH --job-name=scarlet-match
#SBATCH --account=nbody
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/lisasep/scarlet_matched_reproduction_%A_%a.log

set -euo pipefail

scarlet_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep
lisasep_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/lisasep
cd "${scarlet_root}"
export PYTHONPATH="${scarlet_root}:${lisasep_root}/src"
export MPLCONFIGDIR=/tmp/scarlet-match-mpl-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME=/tmp/scarlet-match-cache-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

starts=(A B C)
start="${starts[SLURM_ARRAY_TASK_ID]}"
max_iter="${SCARLET_MAX_ITER:-300}"
run_label="${SCARLET_RUN_LABEL:-matched300}"
fit_dtype="${SCARLET_FIT_DTYPE:-float32}"
channel_chunk_size="${SCARLET_CHANNEL_CHUNK_SIZE:-64}"
output_dir="${lisasep_root}/benchmark_artifacts/collaborator_blend_comparison/scarlet_${run_label}_start${start}"

/nobackup/user/bustam1/lisastack_a6000/jwst/venv-scarlet/bin/python \
  benchmarks/run_collaborator_reproduction.py \
  --data-root /panfs/accrepfs.vampire/nobackup/userspace/bustam1/lisastack_a6000/jwst/collab \
  --output-dir "${output_dir}" \
  --start "${start}" \
  --kernel-size 47 \
  --max-iter "${max_iter}" \
  --relative-tolerance 1e-11 \
  --dtype "${fit_dtype}" \
  --channel-chunk-size "${channel_chunk_size}"
