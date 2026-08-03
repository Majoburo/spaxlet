# Collaborator reproduction

## macOS setup

Use Python 3.11. In Terminal, run:

```bash
git clone --branch scarlet-test --single-branch https://github.com/Majoburo/spaxlet.git
cd spaxlet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyter
```

Put these files together in one data folder:

- `morphology_galaxy_cube_004.fits`
- `nirspec_ifu_PRISM_CLEAR_allwave.cube.fits`

Launch the notebook from the repository root:

```bash
jupyter lab benchmarks/collaborator_reproduction.ipynb
```

Change `DATA_ROOT` in the first code cell, then run all cells. The notebook
runs the declared A, B, and C starts and writes results under
`benchmark_artifacts/collaborator_reproduction`.

The fixed reproduction configuration is variable projection with the
PSF-adjusted centroid constraint, float64 arithmetic, all 940 wavelength
channels, and a maximum of 1500 iterations.

If installation reports missing developer tools, run `xcode-select --install`
and try again.
