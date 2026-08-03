#!/bin/bash
#SBATCH --job-name=spaxlet-planet
#SBATCH --account=nbody
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/lisasep/spaxlet_planet_reproduction_%j.log

# Unlike the galaxy campaign this is a single run, not an A/B/C array: with both
# morphologies pinned at the PSF the model is linear in the spectra, so there is
# no bilinear degeneracy for dispersed starts to explore.

set -euo pipefail

scarlet_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/scarlet-lisasep
lisasep_root=/panfs/accrepfs.vampire/home/bustam1/lisastack_a6000/lisasep
cd "${scarlet_root}"
export PYTHONPATH="${scarlet_root}"
export MPLCONFIGDIR=/tmp/spaxlet-planet-mpl-"${SLURM_JOB_ID}"
export XDG_CACHE_HOME=/tmp/spaxlet-planet-cache-"${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

data_root="${SCARLET_DATA_ROOT:-/panfs/accrepfs.vampire/nobackup/userspace/bustam1/lisastack_a6000/jwst/collab}"
max_iter="${SCARLET_MAX_ITER:-1500}"
kernel_size="${SCARLET_KERNEL_SIZE:-47}"
relative_tolerance="${SCARLET_RELATIVE_TOLERANCE:-1e-9}"
output_dir="${lisasep_root}/benchmark_artifacts/planet/spaxlet"

python="/nobackup/user/bustam1/lisastack_a6000/jwst/venv-scarlet/bin/python"

"${python}" -m benchmarks.run_planet_reproduction \
  --data-root "${data_root}" \
  --output-dir "${output_dir}" \
  --kernel-size "${kernel_size}" \
  --max-iter "${max_iter}" \
  --relative-tolerance "${relative_tolerance}"

"${python}" -m benchmarks.plot_planet \
  --product "${output_dir}/spaxlet_planet_recovery.npz" \
  --output-dir "${output_dir}"
