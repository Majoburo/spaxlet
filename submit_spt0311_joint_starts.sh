#!/bin/bash
#SBATCH --job-name=spt0311-joint
#SBATCH --account=nbody
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep/benchmark_artifacts/spt0311_joint_abc_%A_%a.log

set -euo pipefail

project_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep
python_executable=/nobackup/user/bustam1/lisastack_a6000/jwst/venv-scarlet/bin/python
cd "${project_root}"
export PYTHONPATH="${project_root}"
export MPLCONFIGDIR=/tmp/spt0311-joint-mpl-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME=/tmp/spt0311-joint-cache-"${SLURM_ARRAY_JOB_ID}"-"${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

starts=(A B C)
start="${starts[SLURM_ARRAY_TASK_ID]}"
max_iter="${SPT0311_MAX_ITER:-1500}"
output_dir="${project_root}/benchmark_artifacts/spt0311_joint_validated_start${start}"

"${python_executable}" -m benchmarks.run_spt0311_joint_deblend \
  --output-dir "${output_dir}" \
  --sources lens,lz1,lz2,lz3,E,W,C1,C2,C3,L1,L2,L3,L4,L5,L6 \
  --start "${start}" \
  --max-iter "${max_iter}" \
  --relative-tolerance 0 \
  --optimality-tolerance 1e-4 \
  --optimality-check-interval 25 \
  --channel-chunk-size 64 \
  --dtype float32

"${python_executable}" -m benchmarks.plot_spt0311_joint_deblend \
  --product "${output_dir}/spt0311_joint_deblend.npz" \
  --spectra "${output_dir}/spt0311_joint_spectra.fits" \
  --output-dir "${output_dir}/plots"
