# lisasep IFU parity harness

The fixtures in `ifu_parity_contracts.py` pin the 17x19 operator cases and the
six-channel end-to-end feature matrix before production behavior is changed.
They include a compatible null, a close smooth blend, and a deliberately
clumpy negative control.

`ifu_parity_metrics.py` is the shared scoring boundary. It reports
scale-sensitive integrated and binned spectral errors, unit-flux morphology
errors (including centroid and one-/two-pixel structured residuals), observable
whitened residual diagnostics, and pairwise start sensitivity. Both codes must
pass plain arrays through these functions; code-specific scoring is not used
for promotion decisions.

Run Scarlet's local contract tests:

```bash
python -m unittest discover -s tests -p 'test_ifu_parity_*.py'
```

Run both repositories through the existing comparison drivers:

```bash
python benchmarks/run_lisasep_parity.py \
  --lisasep /path/to/lisasep \
  --python /path/to/venv-spaxlet/bin/python
```

Use `--operator-only` for the sub-second projection gate. The complete command
uses all six generated channels; it never strides the 940-channel collaborator
cube. JSON results are written under `benchmark_artifacts/ifu_parity` by
default.

The runner records the tested repository commits and warns when lisasep HEAD
differs from the pinned reference. Truth-referenced scores are diagnostics,
not a run-selection policy.

## Matched positivity reproduction

`compare_matched_positivity.py` removes two confounders in the original
end-to-end matrix: it gives both codes identical raw factor starts and uses a
per-channel delta PSF in Scarlet's model frame. The latter matters because a
`GaussianPSF(sigma=0.3)` model frame changes the declared forward operator by
about 6.6% on the compact mocks; the delta-frame renderer matches lisasep at
about `2e-8` relative L2.

The historical positivity matrix remains the default. Pass
`--feature symmetry` to apply the same opt-in finite-support centered-symmetry,
positivity, and center-revival chain to both codes using the predeclared source
centers. The clumpy case remains a required negative control; a good fit to the
smooth compatible blend cannot by itself promote symmetry as a default.

The same harness exposes Scarlet's existing monotonic operators as
`--feature monotonic-flat`, `monotonic-angle`, and `monotonic-nearest`. Each
uses the declared fixed center, zero forced radial gradient, positivity, and
center revival. These are comparison arms, not new source defaults.

## Exact constraint intersections

`DykstraConstraintChain` is an opt-in closest-point projector for intersections
of constraints that explicitly declare exact convex Euclidean projections.
`CentroidConstraint` supplies the linear fixed-centroid projector; combining it
with positivity gives the closest non-negative morphology at a declared
centroid. The exact chain deliberately rejects heuristic monotonicity and
relaxed symmetry, and raises instead of silently returning a non-converged
iterate. Historical `ConstraintChain` behavior is unchanged.

Pass `--feature centroid` to exercise this exact intersection through the
matched six-channel matrix in both codes. The arm fixes only the catalogued
centroid and positivity; it does not impose symmetry or a radial profile.

Dynamic `ImageMorphology` growth and shrinkage rebase explicit local constraint
centers by the opposite bounding-box shift, preserving their global pixel
coordinates. Coordinate-free historical constraints are reused unchanged.

IFU `Frame` and `Observation` objects can opt into a physical spectral-grid
contract with unit-bearing `wavelengths`. Matching then requires both sides to
declare wavelengths and verifies the channel-mapped grids after unit
conversion. Omitting wavelengths preserves historical broadband behavior.

`Observation.from_ifu_arrays` ingests science, physical wavelengths, measured
variance, and optional integer DQ arrays without changing the legacy
constructor. Non-finite data, non-positive/non-finite variance, selected DQ
bits, and full detector-gap channels receive zero inverse variance and finite
zero-filled data, with mask counts retained on the observation.

`SpatiallyVaryingConvolutionRenderer` is an opt-in response for aligned IFU
frames when independent calibration supplies a rectangular grid of field PSFs.
It requires an intrinsic `DeltaPSF` model frame and explicit
`(channel, anchor, y, x)` kernels plus row/column anchor coordinates. Bilinear
source-plane weights form a partition of unity, full and channel-chunked
rendering share the same cached-FFT operator, and autograd uses its registered
exact adjoint. Ordinary `Observation.match` behavior is unchanged.

`run_collaborator_reproduction.py` applies the same correction to the complete
940-channel A/B/C experiment. It uses the same catalog-centered blobs,
crop-then-recenter PSFs, measured variance, source labels, and 300-iteration
budget as the corrected lisasep comparison. The batch driver defaults to a
single-precision fit and 64-channel likelihood chunks. On the 940x47x47 cube,
these settings reduced measured one-step peak RSS from 1.43 GB to 0.615 GB;
the strict recovery metrics were unchanged in the float32/chunk parity checks.
The Python API retains float64 and unchunked defaults for compatibility.
Submit all predeclared starts with:

```bash
sbatch submit_collaborator_reproduction.sh
```

For a predeclared time-to-equal-fit comparison with a larger iteration cap,
keep the original products isolated by setting both batch variables:

```bash
sbatch --export=ALL,SCARLET_MAX_ITER=1500,SCARLET_RUN_LABEL=matched1500 \
  submit_collaborator_reproduction.sh
```

Override the bounded-memory settings with `SCARLET_FIT_DTYPE` and
`SCARLET_CHANNEL_CHUNK_SIZE`, or select a declared adaprox scheme with
`SCARLET_OPTIMIZER_SCHEME`. Pass `--profile-memory` directly to the Python
driver to print current and peak RSS checkpoints. Every reproduction reports
joint, spectral, and scale-gauge-quotiented morphology proximal-gradient
residuals; a small loss change alone is not treated as convergence evidence.
The batch driver checks the joint residual every 100 iterations and stops at
`1e-4` by default. Override those gates with
`SCARLET_OPTIMALITY_TOLERANCE` and `SCARLET_OPTIMALITY_CHECK_INTERVAL`.
Free raw factors are feasibility-projected and placed in a common L1
morphology gauge before fitting; this is an exact no-op on the unit-sum A/B/C
starts and removes arbitrary caller-supplied spectrum/morphology rescalings.
Saved metrics and NPZ products also contain exact two-sided pairwise mixing
intervals plus integrated-spectrum and unit-flux-morphology envelopes. These
are labeled structural sensitivity floors, not posterior or `+/-1 sigma`
uncertainties.

The full-cube driver retains free positivity factors by default. Set
`SCARLET_FEATURE=centroid` in the batch environment, or pass
`--feature centroid` directly, to apply the previously gated exact
non-negative fixed-centroid constraint at the two independently declared
catalog positions. Always use a separate run label and submit all A/B/C
starts; this conditional arm is not a new default.

Plot a saved Scarlet product without importing lisasep:

```bash
python -m benchmarks.plot_collaborator_reproduction \
  --product /path/to/scarlet_matched_startA.npz \
  --output-dir /path/to/plots
```

The metrics JSON is discovered beside the NPZ by default and records the truth
FITS path. The command writes spectra with exact structural envelopes, unit-flux
morphologies and residuals, the collapsed whitened data-minus-model residual,
and a table of truth-independent fit diagnostics. Use `--metrics` or `--truth`
to override either discovered path.

`collaborator_reproduction.ipynb` wraps the same two module commands for
interactive review. Its first code cell is the only configuration surface;
the fit and plotting logic remain in the tested Python modules.

## SPT0311-58 public-cube deblend

`run_spt0311_deblend.py` is a truth-independent, many-source test on the public
MAST NIRSpec IFU cubes associated with arXiv:2312.00899. It registers the
paper's relative source catalog through the foreground lens, extracts a
wavelength-dependent empirical PSF from standard star 1808347, calibrates the
blank-sky level and ERR scale, and simultaneously fits one positive
spectrum-morphology factor for each catalog component. Finite local morphology
supports prevent a fixed centroid from being satisfied by unphysical distant
lobes. Those support sizes and the SPT0311 catalog are benchmark choices, not
Scarlet defaults.

Reusable operations discovered while building the test live in core:
`spaxlet.empirical_psf_kernels`, `spaxlet.estimate_ifu_background`, and
`spaxlet.measure.factorization`. Their focused tests do not depend on the
downloaded JWST files.

Example broad PRISM run (15 components):

```bash
python benchmarks/run_spt0311_deblend.py \
  --cube data/spt0311_mast/jw01264-o013_t010_nirspec_prism-clear_s3d.fits \
  --psf-cube data/spt0311_mast/calibration_star_1808347/jw01128-o009_t007_nirspec_prism-clear_s3d.fits \
  --output-dir benchmark_artifacts/spt0311_prism_many_source \
  --mode prism --wavelength-min 2.85 --wavelength-max 5.25 \
  --max-iter 25 --channel-chunk-size 64
```

Example G395H H-beta/[O III] run (16 components, including L7):

```bash
python benchmarks/run_spt0311_deblend.py \
  --cube data/spt0311_mast/jw01264-o013_t010_nirspec_g395h-f290lp_s3d.fits \
  --psf-cube data/spt0311_mast/calibration_star_1808347/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits \
  --output-dir benchmark_artifacts/spt0311_g395h_oiii_many_source \
  --mode g395h --wavelength-min 3.75 --wavelength-max 4.02 \
  --max-iter 20 --channel-chunk-size 64
```

Each run writes a FITS spectral table, a compressed NPZ containing normalized
morphologies/model/residuals, and a JSON report with mask, PSF, noise, fit, and
provenance diagnostics. Plot a product with:

```bash
python benchmarks/plot_spt0311_deblend.py \
  --product benchmark_artifacts/spt0311_g395h_oiii_many_source/spt0311_deblend.npz \
  --spectra benchmark_artifacts/spt0311_g395h_oiii_many_source/spt0311_deblended_spectra.fits \
  --output benchmark_artifacts/spt0311_g395h_oiii_many_source/diagnostic.png
```

These are archive-pipeline pilots, not measurements from the paper's custom
0.05-arcsec reduction. The finite iteration budgets also leave morphology
projected-gradient residuals around `1e-2`; use larger budgets and stability
tests before treating extracted line fluxes as final science products.

The real-cube runner defaults to joint constrained NMF (`adaprox`/AMSGrad),
with a nonnegative tabulated spectrum outer-producted with a nonnegative local
morphology for every component. A G395H constraint ladder is available through
`--morphology-constraint`: `positivity`, exact `centroid`, `monotonic`,
`symmetry`, and benchmark-selected hybrid arms. On the 3.75--4.02 um pilot,
positivity alone gives the lowest chi-square but allows nine source identities
to move by more than one spaxel. Exact centroid plus positivity costs only
0.36% in chi-square and eliminates every such drift, so it is the selected
default. Global monotonicity and symmetry are rejected by the comparison.

`compare_spt0311_constraints.py` records fit, source-drift, morphology-area,
180-degree-asymmetry, and spectrum-stability diagnostics for precomputed arms.
The first supplied arm is the spectral and symmetry-selection reference. The
diagnostic plot uses a shared numerical MJy/sr color scale for data, model, and
residual, with labeled colorbars and cube-level RMS values; its whitened panel
retains a separate standardized scale.
