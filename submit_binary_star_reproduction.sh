#!/bin/bash
#SBATCH --job-name=spaxlet-binary
#SBATCH --account=nbody
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep/benchmark_artifacts/binary_star/spaxlet_binary_%j.log

set -euo pipefail

scarlet_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep
artifact_root="${scarlet_root}/benchmark_artifacts/binary_star"
cube="${artifact_root}/input/majo_blended_binary_star_cube.fits"
psf=/panfs/accrepfs.vampire/nobackup/userspace/bustam1/lisastack_a6000/jwst/collab/nirspec_ifu_PRISM_CLEAR_allwave.cube.fits
python=/nobackup/user/bustam1/lisastack_a6000/jwst/venv-scarlet/bin/python

cd "${scarlet_root}"
export PYTHONPATH="${scarlet_root}"
export MPLCONFIGDIR=/tmp/spaxlet-binary-mpl-"${SLURM_JOB_ID}"
export XDG_CACHE_HOME=/tmp/spaxlet-binary-cache-"${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${python}" -m benchmarks.run_binary_star_reproduction \
  --cube "${cube}" \
  --psf "${psf}" \
  --output-dir "${artifact_root}"

"${python}" -m benchmarks.plot_planet \
  --product "${artifact_root}/binary_star_recovery.npz" \
  --report "${artifact_root}/binary_star_report.json" \
  --output-dir "${artifact_root}" \
  --filename binary_star_spectra.png
