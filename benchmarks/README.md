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

The generated regression analogue in `ifu_parity_contracts.py` has nine
overlapping rank-one sources and, by default, 1,024 spectral slices. The slice
count is configurable: the contract is joint recovery from a broad color stack,
not recovery at exactly 1,024 channels. Two sources have
sub-RMS peak signal in a typical individual slice, two overlapping sources
have deliberately similar spectra, and the catalog includes compact,
asymmetric, and wide morphologies. The framework-neutral forward fixture also
has a channel-dependent PSF, seeded noise, and optional nonuniform variance and
masks.

`run_synthetic_many_source_ifu.py` always fits one constrained 2D morphology
per source jointly with its full-length spectrum. Spectral starts come from
noisy apertures rather than truth. The recovery ladder uses a delta PSF to
isolate the joint factorization; a separate assertion renders the same truth
through the channel-dependent PSF with Spaxlet. At the 60-iteration
reproducibility budget,
selective symmetry gives the best joint residual (`chi2/N = 1.0023`), followed
by centroid (`1.0090`), positivity (`1.0226`), and global symmetry (`1.2935`).
With centroid constraints, truncated, correct, and oversized supports give
`1.1888`, `1.0090`, and `1.0184`, respectively. The complete color stack also
reduces mean morphology error for the two per-slice-faint sources by 18%
relative to 16 evenly spaced colors.

The focused unit test uses a shorter fixed budget while preserving those arm
orderings. Run it without archive data:

```bash
python -m unittest discover -s tests -p 'test_ifu_many_source.py'
```

Run one full residual arm directly with:

```bash
python -m benchmarks.run_synthetic_many_source_ifu \
  --slices 1024 --support correct --constraint selective_symmetry \
  --start 0 --max-iter 60
```

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

The synthetic finding was then tested on the broad useful G395H interval,
2.90--5.15 um: all 3,384 selected detector slices were fitted jointly rather
than forcing an arbitrary channel count. At 30 iterations, positivity,
centroid, and selected-source symmetry give `chi2/N = 1.2353`, `1.2359`, and
`1.2587`. Padding every morphology support by two pixels per side worsens the
positivity and hybrid values to `1.2614` and `1.2899`; neither synthetic
selective symmetry nor larger supports transfer to this cube. At 100
iterations, positivity reaches `1.2089` and centroid `1.2111`, a 0.18% cost for
fixing the catalog identity: positivity leaves 11/16 sources more than one
spaxel from their catalog positions, versus 0/16 for centroid. Thus positivity
is the best residual-only arm, while centroid remains the safer default for
source-attributed spectra. The median physical data and residual RMS are both
3.61 MJy/sr, but the weighted residual is 1.10 measured RMS; only 3.8% of valid
voxels have fitted model amplitude above one local RMS. Coherent structure in
the collapsed whitened residual shows that the remaining excess is not removed
by changing only source morphology constraints.

Reproduce the broad comparison with separate output directories, for example:

```bash
python benchmarks/run_spt0311_deblend.py \
  --cube data/spt0311_mast/jw01264-o013_t010_nirspec_g395h-f290lp_s3d.fits \
  --psf-cube data/spt0311_mast/calibration_star_1808347/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits \
  --output-dir benchmark_artifacts/spt0311_g395h_broad_centroid \
  --mode g395h --wavelength-min 2.90 --wavelength-max 5.15 \
  --morphology-constraint centroid --max-iter 100 --channel-chunk-size 64
```

Use `--support-padding 2` only as a comparison arm; zero is the benchmark
default and was favored by the broad-cube pilots.

## Spectral smoothness arm

`SpectralSmoothnessConstraint` is a reusable one-dimensional proximal penalty
for non-negative spectra. It penalizes second divided differences on the
physical wavelength grid, detects large wavelength gaps, and uses proximal
Dykstra iterations to combine quadratic curvature with positivity. A fixed
per-spectrum reference amplitude makes its public strength dimensionless and
the proximal response invariant to an overall spectral rescaling.

The focused contract deliberately distinguishes spectral resolution from
plotting smoothness. At its declared test setting, a roughly five-channel
Gaussian line retains 95.4% of its peak and all integrated flux, while a
one-channel spike is reduced to 44.8% of its peak. The complete nine-source,
1,024-channel joint fit also gates the selected synthetic arm: dimensionless
strength 50 improves mean spectral-shape cosine from 0.9832 to 0.9911, reduces
mean curvature by 31%, and changes `chi2/N` by 0.000815.

On the broad real G395H cube, an amplitude-normalized strength ladder rejects
3 and selects 0.1 as the conservative diagnostic arm. At 100 iterations,
strength 0.1 changes `chi2/N` from 1.21110 to 1.21205 (0.078%), reduces median
normalized spectral curvature by 32%, and preserves the resolved lens feature
at 3.8153 um to 99.94% in peak and 101.3% in nine-channel excess. The isolated
4.9977 um lens spike falls to 83.7% in peak. The weakest source's integrated
fitted flux changes by 19.5%, and the spectral projected-gradient residual is
still 0.256, so this arm remains opt-in rather than becoming the default.

Run the real-cube arm by adding:

```bash
--spectral-smoothness-strength 0.1
```

The unsmoothed spectrum is always saved at native sampling; diagnostic
smoothing is not substituted for the fitted model. The overview plot now
includes the foreground lens by default.

The corresponding known-truth plot is reproducible with:

```bash
python -m benchmarks.plot_synthetic_spectral_smoothness \
  --output-dir benchmark_artifacts/synthetic_spectral_smoothness \
  --strengths 0,50,300 --slices 1024 --max-iter 40 \
  --constraint selective_symmetry --support correct
```

It writes native truth-versus-fit spectra, unit-flux morphologies, and
data/model/residual panels on one common display scale, plus numerical metrics
in `comparison.json`. In this selective-symmetry arm, strengths 0, 50, and 300
give mean spectral cosines 0.9751, 0.9874, and 0.9902, while `chi2/N` changes
from 1.0433 to 1.0442 and 1.0467. Stronger smoothing therefore denoises
spectral shape, especially for the two sources below one peak-spaxel S/N per
channel, but does not cure flux mixing between overlapping bright sources.
The reporting regression fixes every morphology to unit sum and verifies that
the reported spectra and morphologies reconstruct the fitted cube exactly.

This differs from the successful local NGC 7469 fixed-count case in `lisasep`.
That simpler fit uses one fixed chromatic point source and one learned
full-support host, jointly profiles both spectra, and selected zero host
smoothness. Its clean nuclear spectrum primarily reflects high per-channel S/N
and strong spatial identifiability, not a spectral smoothing operation.

## Joint spectral and spatial continuity

`SpatialSmoothnessConstraint` is the morphology analogue: an exact quadratic
nearest-neighbor penalty, implemented as a sparse screened-Poisson proximal
solve. `ProximalDykstraConstraintChain` combines it with positivity and exact
centroid or symmetry constraints as one order-independent convex proximal map.
It is available in the real runner through `--spatial-smoothness-strength` and
is disabled by default.

The four-arm synthetic ablation is:

```bash
python -m benchmarks.plot_synthetic_joint_smoothness \
  --output-dir benchmark_artifacts/synthetic_joint_smoothness \
  --spectral-strength 300 --spatial-strength 300 \
  --slices 1024 --max-iter 40 --constraint selective_symmetry
```

Neither, spectral-only, spatial-only, and joint arms give `chi2/N` 1.0433,
1.0467, 1.0389, and 1.0424. Spatial smoothness reduces the mean morphology
error of the seven detected sources from 0.1484 to 0.1442 and the mean
scale-sensitive spectral error from 0.5452 to 0.5027. However, it increases
the two sub-unity sources' morphology error from 0.9648 to 0.9866; the combined
arm reaches 1.1085. Spectral continuity improves their mean spectral cosine
from 0.8888 to about 0.957 but does not restore spatial identity. Thus neither
penalty is a universal default: spectral strength should reflect resolvable
line width, while spatial strength should be applied preferentially to broad,
well-detected morphologies rather than compact faint components.

The same ordering was checked on the 3,384-slice real G395H pilot. Uniform
spatial strengths 100 and 1,000 give `chi2/N` 1.23634 and 1.23760 versus the
unregularized 1.23595. Strength 100 reduces median morphology roughness to
52.8% of baseline with median integrated-flux movement 2.2% and maximum 9.6%;
strength 1,000 reduces roughness to 15.1% but moves a weak source by 28.7%.
The conservative combined arm, spectral 0.1 plus spatial 100, gives
`chi2/N = 1.23652`, roughness ratio 0.514, curvature ratio 0.728, median flux
movement 6.6%, and maximum movement 21.5%. Its spectral projected-gradient
residual is 0.471 after 30 iterations, so it is explicitly rejected as a
converged science product. These pilots retain both penalties as opt-in core
infrastructure but do not promote either to the real-run default.

`compare_spt0311_constraints.py` records fit, source-drift, morphology-area,
180-degree-asymmetry, and spectrum-stability diagnostics for precomputed arms.
The first supplied arm is the spectral and symmetry-selection reference. The
diagnostic plot uses a shared numerical MJy/sr color scale for data, model, and
residual, with labeled colorbars and cube-level RMS values; its whitened panel
retains a separate standardized scale.
