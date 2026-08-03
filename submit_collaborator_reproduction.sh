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
export PYTHONPATH="${scarlet_root}"
export MPLCONFIGDIR=/tmp/scarlet-match-mpl-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME=/tmp/scarlet-match-cache-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

starts=(A B C)
start="${starts[SLURM_ARRAY_TASK_ID]}"
max_iter="${SCARLET_MAX_ITER:-1500}"
run_label="${SCARLET_RUN_LABEL:-varproj_psfcentroid1500}"
fit_dtype="${SCARLET_FIT_DTYPE:-float64}"
channel_chunk_size="${SCARLET_CHANNEL_CHUNK_SIZE:-64}"
optimizer_scheme="${SCARLET_OPTIMIZER_SCHEME:-amsgrad}"
optimizer="${SCARLET_OPTIMIZER:-variable_projection}"
minimum_volume_strength="${SCARLET_MINIMUM_VOLUME_STRENGTH:-0}"
spectral_max_iter="${SCARLET_SPECTRAL_MAX_ITER:-100}"
spectral_tolerance="${SCARLET_SPECTRAL_TOLERANCE:-1e-8}"
feature="${SCARLET_FEATURE:-centroid_psf}"
optimality_tolerance="${SCARLET_OPTIMALITY_TOLERANCE:-1e-4}"
optimality_check_interval="${SCARLET_OPTIMALITY_CHECK_INTERVAL:-20}"
output_dir="${lisasep_root}/benchmark_artifacts/collaborator_blend_comparison/scarlet_${run_label}_start${start}"

/nobackup/user/bustam1/lisastack_a6000/jwst/venv-scarlet/bin/python \
  -m benchmarks.run_collaborator_reproduction \
  --data-root /panfs/accrepfs.vampire/nobackup/userspace/bustam1/lisastack_a6000/jwst/collab \
  --output-dir "${output_dir}" \
  --start "${start}" \
  --kernel-size 47 \
  --max-iter "${max_iter}" \
  --relative-tolerance 1e-11 \
  --dtype "${fit_dtype}" \
  --channel-chunk-size "${channel_chunk_size}" \
  --optimizer "${optimizer}" \
  --optimizer-scheme "${optimizer_scheme}" \
  --minimum-volume-strength "${minimum_volume_strength}" \
  --spectral-max-iter "${spectral_max_iter}" \
  --spectral-tolerance "${spectral_tolerance}" \
  --feature "${feature}" \
  --optimality-tolerance "${optimality_tolerance}" \
  --optimality-check-interval "${optimality_check_interval}"
