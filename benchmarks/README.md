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

The real-cube likelihood is a fixed, heteroscedastic Gaussian with variance
`(JWST S3D ERR * empirical noise scale)^2`. Thus source shot noise contributes
only to the extent that the JWST pipeline propagated it into `ERR`; the fit
does not update a Poisson variance from its current source model. This avoids
letting a brighter trial model reduce its own weighted residual penalty and is
appropriate for these faint, background-dominated cubes.

The joint runner also applies the known redshift inventory: `E`, `W`,
`C1`--`C3`, and `L1`--`L6` are constrained to zero below a conservative
0.95-um observed Lyman-break edge, while the z=1.0343 lens and the three
z<3 `lz` sources remain unconstrained there. This prevents high-redshift
factors from borrowing the foreground lens's short-wavelength continuum.
Use `--disable-high-redshift-support` only for the unconstrained control.

`run_spt0311_joint_deblend.py` fits the PRISM and G395H cubes in one
likelihood. Its default is genuinely shared: every source has one morphology
and one latent spectrum. G395H selects the fine latent samples in its range,
while a fixed sparse `SpectralResponse` integrates those same samples into
the wider PRISM wavelength bins. Across the overlap, the default also applies
a Gaussian line-spread response whose wavelength-dependent FWHM comes from
the public STScI ETC PRISM resolving-power table; use
`--prism-line-response none` for the top-hat-only control. PRISM-only
wavelengths remain on the native PRISM grid. `--spectral-model independent`
retains two unconstrained spectral blocks only as an explicit control. The
reusable response constructors are `spaxlet.binned_spectral_response` and
`spaxlet.gaussian_spectral_response`; both preserve constant flux density and
have a differentiable channel-chunked rendering path.

Both real-cube drivers also mask compact positive or negative islands that
survive the archive cube builder's DQ flags. The default requires a 12-sigma
seed, grows its same-sign footprint to 3 sigma, and masks only footprints of
at most 12 spatial pixels in one wavelength slice. This removes detector-level
outliers without clipping resolved PSF-supported emission; every count and
threshold is recorded in the JSON report. Use `--disable-outlier-mask` for the
unfiltered control.

On the complete 941-channel PRISM plus 3,610-channel G395H cubes, the default
15-factor run masks 452 and 837 voxels, respectively (0.030% and 0.015%, or
less than 0.02% of the combined valid likelihood). With the physical
high-redshift support, at 30 iterations it reaches joint
`chi2/N = 1.24235`, with PRISM at `1.41400` and G395H at `1.19600`. The
otherwise identical unconstrained-blue control reaches `1.24200`; the 0.028%
cost removes physically impossible z~6.9 flux below the Lyman break but does
not by itself resolve foreground-lens spatial cross-talk. The top-hat
control gives `1.24547`, `1.41958`, and `1.19846`, so the calibrated PRISM LSF
is preferred independently of the outlier correction. A cube-builder artifact
at 4.98237 um that previously produced a one-channel 158.7 uJy `lz2` amplitude
falls to 1.07 uJy. The sparse response/adjoint completes this run in 90.9 s
with 64-channel chunks; the equivalent gather implementation did not finish
after 27 minutes because it materialized the latent support at every spaxel.

The 15-factor result remains attribution-limited where the foreground lens
overlaps the high-redshift system. In the unconstrained morphology arm, `W`
places 62% of its unit-flux morphology within five spaxels of the lens center;
the two factors have morphology and spectrum cosines of 0.80 and 0.86. `L4`
also breaks into four islands above 20% of its fitted peak. A matched control
that makes `W`, `L4`, and `E` monotonic removes the disconnected islands but
moves their centroids by 2.32, 1.30, and 0.71 spaxels and increases lens-region
`W` flux to 70%, so it is rejected. These extracted component spectra should
not be treated as final measurements without a line-map or external-image
foreground-lens template.

The known-truth two-galaxy protocol is carried over as a numerical gate rather
than a new morphology guess. `--start A` uses the catalog widths; start B makes
the foreground factors broad and z~6.9 factors narrow, while C exchanges those
widths. Run all three in separate output directories and accept source
attribution only when independent starts reach the same basin. Set
`--optimality-tolerance 1e-4` (with a sufficiently large `--max-iter`) to use
the strict numerical reference from the two-galaxy campaign. That value is a
stationarity diagnostic, not a universal real-data science threshold: a
real-cube stopping rule must show that additional optimization changes the
reported line fluxes or redshifts by much less than their empirically measured
uncertainty. The
joint PRISM sparse response mixes many latent wavelengths into one observed
channel, so the channel-selection variable-projection implementation is not
used here; the report records that limitation explicitly. Minimum-volume
regularization remains off because it did not rescue the synthetic bad basin.

`validate_spt0311_joint_starts.py` therefore fails closed without inventing
thresholds from the idealized simulation. Convergence and identical input/model
provenance are hard numerical gates. It reports rendered-model start spread,
per-source spectral and morphology spread, pairwise factor collisions, and
spatial lag-one structure in the collapsed whitened residuals. It cannot mark
a result science-ready unless supplied a calibration JSON explicitly labeled
`real_residual_injection`; its scene and source-specific thresholds must come
from sources injected into the actual correlated cube residuals. This allows a
scene model to be numerically usable while keeping individual confused sources
unpublished.

The optimizer checkpoints also track each source's integrated spectrum,
effective morphology area, centroid, and O II/H-beta/O III/H-alpha window
integrals. Validation requires at least two late checkpoints and compares their
drift against the injection-derived limits, preventing a small projected
gradient from masking observables that are still moving. Build each injected
cube with `make_spt0311_residual_injection.py`, fit its A/B/C starts using
`--data-override`, validate those starts, and combine the campaign validations
with `calibrate_spt0311_residual_injections.py`. That final calibration marks a
source non-reportable when its injected spectrum shape, morphology, or
integrated or science-window flux fails the explicitly supplied truth
tolerances; it does not inflate a stability threshold to make a failed
recovery pass. A minimal one-realization calibration flow is:

```bash
python -m benchmarks.make_spt0311_residual_injection \
  --product BASELINE/spt0311_joint_deblend.npz \
  --output INJECTIONS/seed01.npz --seed 1

# Fit seed01.npz from A, B, and C with the same options as BASELINE, adding:
#   --data-override INJECTIONS/seed01.npz --start START

python -m benchmarks.validate_spt0311_joint_starts \
  --run A=INJECTION_A --run B=INJECTION_B --run C=INJECTION_C \
  --output INJECTIONS/seed01_validation.json
python -m benchmarks.calibrate_spt0311_residual_injections \
  --baseline-dir BASELINE \
  --validation INJECTIONS/seed01_validation.json \
  --output INJECTIONS/calibration.json
python -m benchmarks.validate_spt0311_joint_starts \
  --run A=REAL_A --run B=REAL_B --run C=REAL_C \
  --calibration INJECTIONS/calibration.json \
  --output REAL_A/validation.json --fail-on-rejection
```

Use multiple `--validation` arguments from independent seeds for a production
calibration; one realization is only a plumbing check.

```bash
python -m benchmarks.run_spt0311_joint_deblend \
  --output-dir benchmark_artifacts/spt0311_joint \
  --sources lens,lz1,lz2,lz3,E,W,C1,C2,C3,L1,L2,L3,L4,L5,L6
python -m benchmarks.plot_spt0311_joint_deblend \
  --product benchmark_artifacts/spt0311_joint/spt0311_joint_deblend.npz \
  --spectra benchmark_artifacts/spt0311_joint/spt0311_joint_spectra.fits \
  --output-dir benchmark_artifacts/spt0311_joint/plots
```

The two residual figures use one identical surface-brightness scale for data,
model, and residual, and print data/model/residual RMS plus residual/data RMS
and chi-square per valid voxel. The source gallery shows every morphology and
both instrument-space views of the shared spectrum, including the lens.

The generated regression analogue in `ifu_parity_contracts.py` is retained as
a **stress fixture**, not a source-recovery certificate. It has nine
overlapping rank-one sources and, by default, 1,024 spectral slices. The slice
count is configurable: the contract is joint recovery from a broad color stack,
not recovery at exactly 1,024 channels. Two sources deliberately have
sub-RMS peak signal in a typical individual slice, two overlapping sources
have deliberately similar spectra, and the catalog includes compact,
asymmetric, and wide morphologies. The framework-neutral forward fixture also
has a channel-dependent PSF, seeded noise, and optional nonuniform variance and
masks. Its global residual and arm-ordering assertions must not be interpreted
as proof that all nine factors were recovered.

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

### Many-source recovery contract

`many_source_recovery_contract.py` supplies the complementary positive
control. It contains ten recoverable sources, including a broad central factor,
elliptical and clumpy morphologies, overlapping finite supports, distinct line
complexes, a wavelength-dependent PSF, heterogeneous read-plus-source
variance, masked voxels, and a seeded residual field with both spectral and
spatial covariance. The fit intentionally uses only the diagonal variance, so
the recovery gate remains valid under modest likelihood misspecification.

`run_synthetic_many_source_recovery.py` fits the observed, PSF-convolved cube
with constrained variable projection from broad, narrow, and very broad
starts. Note that the fit is **conditional on an oracle catalog**: the source
count, every `spec.center`, every `spec.support` box, and each centroid
constraint target are taken from the truth specification, the last computed
from the truth morphology itself. The reported centroid errors near `1e-11`
are that constraint being enforced, not a recovered position. Treat the
declaration as "given a perfect catalog, the factorization is recoverable",
which is what makes the signed diagnostics below interpretable.

The test requires **every source in every start** to satisfy spectrum,
morphology, centroid, integrated-flux, identity, stationarity, and residual
gates. At the declared 180-iteration budget, all spectral cosines exceed
0.999, all morphology cosines exceed 0.975, integrated-flux errors are below
8.5%, and morphology relative errors are below 22%.

### Signed per-source errors and the flux transfer

The gates above are absolute, and an absolute error cannot distinguish ten
independent errors from one redistribution of flux between overlapping
factors. `signed_recovery_metrics.py` keeps the sign, splits the error into
continuum and line parts on the declared line windows, and reports a
**cancellation ratio** `|sum d| / sum |d|` over the per-source absolute flux
differences. The data constrain the summed cube, so a redistribution leaves
the scene total nearly unchanged; independent errors would give a ratio
near one.

The measured ratio is 0.017 to 0.074 across three starts and four noise
seeds. The errors are one redistribution. Its direction is fixed: `lens`, the
broad central factor every other source overlaps, gains `+5.4% +- 1.7%`,
positive in every run and in all eight wavelength bins, while the other nine
sources lose flux. The surplus is achromatic, so it is a flux transfer rather
than a spectral-shape error and is not reduced by adding channels. The
recorded `integrated_flux_relative_error` of 7.6% for `lens` sat inside the
8.5% gate and read as noise; it is not noise.

Two controls separate the causes. On the **noiseless** cube the maximum signed
error falls 5.96%, 1.86%, 0.31%, 0.024% at 180, 600, 2,000 and 6,000
iterations, so the model plus the oracle catalog do identify every source and
the direction is not flat. On the **noisy** cube the same sweep gives 7.64%,
6.06%, 6.20%, 6.26% while `relative_projected_gradient` falls to `1.70e-06`:
roughly 1.4 points of the pilot number is an optimization transient and the
remaining 6.3% is a converged estimation bias. The broad
central component is the slowest-converging direction, so a truncated fit
parks additional flux there on top of that bias.

Consequently the 180-iteration budget is a **pilot**, not a converged result,
and its `relative_projected_gradient` gate of `4e-4` does not certify the flux
numbers: the converged value is `1.19e-05`, about two orders of magnitude
lower. `RECOVERY_PILOT_MAX_ITER` and `RECOVERY_CONVERGED_MAX_ITER` name the two
budgets. Quote converged-budget results for anything reporting a per-source
flux. The constraint ablation and smoothness ladders were all measured at the
pilot budget and may therefore rank convergence rate rather than recovery;
they have not yet been re-measured.

```bash
python -m benchmarks.diagnose_many_source_transfer \
  --starts A,B,C --seeds 8675309 --max-iter 180
SPAXLET_CONVERGED_RECOVERY=1 python -m unittest discover \
  -s tests -p 'test_ifu_many_source_recovery.py'
``` Selecting the minimum
chi-square start without consulting truth also passes the tighter 8%, 21%, and
8% spectrum, morphology, and integrated-flux gates. Two additional correlated
noise seeds retain every source identity and keep maximum spectrum and
integrated-flux errors below 10%, so the declaration is not tied to one noise
draw.

```bash
python -m unittest discover -s tests -p 'test_ifu_many_source_recovery.py'
python -m benchmarks.run_synthetic_many_source_recovery \
  --start A --max-iter 180
python -m benchmarks.plot_synthetic_many_source_recovery \
  --output-dir benchmark_artifacts/many_source_recovery \
  --start A --max-iter 180
```

The diagnostic scene uses absolute `x`/`y` pixel coordinates and labels every
catalog position. Cyan 20/50/80% contours show truth and dashed magenta
contours show the recovered morphology, both on the scene overview and on
source cutouts.

### Detected-catalog arm

`run_synthetic_many_source_detected_catalog.py` removes the oracle
conditioning. It detects sources with `lisasep.segmentation`, derives the
entire catalog -- count, centers, support boxes, centroid targets and start
widths -- from the inverse-variance detection image, fits with the identical
forward model and constraints, and matches to truth only for scoring. It needs
the sibling checkout on the path:

```bash
PYTHONPATH=../lisasep/src python -m \
  benchmarks.run_synthetic_many_source_detected_catalog \
  --max-iter 2000 --dilation 0 \
  --output-dir benchmark_artifacts/many_source_detected_catalog_dil0
```

At the converged budget, with every arm at `relative_projected_gradient`
below `2.7e-05`:

| arm | chi2/N | max abs signed flux | max morphology L2 |
| --- | --- | --- | --- |
| oracle | 0.944663 | 0.0713 | 0.194 |
| detected, dilation 0 | 1.066834 | 0.7282 | 0.616 |
| detected, dilation 2 | 1.318503 | 1.4430 | 1.091 |
| undercount, dilation 0 | 6.284374 | 1.0581 | 1.189 |
| overcount, dilation 0 | 1.063955 | 0.7275 | 0.616 |

Four results follow, and none of them is a budget artifact.

**Detection is not the limitation.** All ten sources are found, at exactly the
correct integer centers: the center offset is `0.000` px for every source. The
count and the positions are recoverable from this cube.

**Support size is the limitation, and it is a knife edge.** Watershed segments
at zero dilation are systematically truncated -- `lens` 9 against a truth
support of 15, `northeast` 5 against 9 -- and the maximum per-source flux error
is 73%, ten times the oracle value. Dilating by two pixels overshoots, giving
supports of 11 against 9, heavy overlap, and a 144% maximum error. The oracle
catalog's real content is therefore the support extents, not the positions,
which agrees with `DEBLENDING_STATE.md` section 7.2 ranking support masks as an
assumption-light constraint that carries real information.

**Chi-square ranks support choices correctly.** Dilation 0 beats dilation 2 on
both the residual (1.0668 against 1.3185) and the truth error (73% against
144%), so support extent is one catalog property that can be tuned without
truth.

**Chi-square detects a missing source but rewards a spurious one.** Dropping
`lens` sends `chi2/N` to 6.28, which no analysis would miss. Adding an
eleventh component in a quiet corner *lowers* the residual, from 1.066834 to
1.063955, while leaving every other source unchanged. Since the residual is
the only model-selection quantity available on a real cube, catalog
completeness is testable there and catalog parsimony is not.

### Joint imaging arm

`many_source_imaging_contract.py` renders the same latent scene into three
broad bands through a narrow, wavelength-independent PSF (`sigma = 0.32` px
against the IFU's 0.62 to 1.12), and
`run_synthetic_many_source_joint_imaging.py` fits both observations against one
model frame. Each observation declares its own channels, so every source keeps
a single morphology and one spectrum spanning IFU channels and imaging bands
together; scoring uses the IFU channels alone. The bands share the IFU pixel
grid on purpose: what constrains a support is PSF width, not sampling, and a
shared grid keeps the comparison with the IFU-only arms exact. A genuinely
finer detector would additionally need real WCS objects and the `ResolutionRenderer`
path, which has no test coverage in this repository.

At the converged budget, every arm below `2e-05` relative projected gradient:

| arm | sources | IFU chi2/N | max abs signed flux | max morphology L2 |
| --- | --- | --- | --- | --- |
| oracle, IFU only | 10 | 0.944663 | 0.0713 | 0.194 |
| detected from IFU, IFU only | 10 | 1.066821 | 0.7281 | 0.616 |
| detected from imaging, IFU only | 13 | 1.181401 | 0.3546 | 0.695 |
| detected from imaging, joint | 13 | 1.231195 | 0.4033 | 0.750 |
| oracle, joint | 10 | 0.947556 | 0.0600 | 0.168 |

**A sharper band does not fix the detected-catalog failure.** The maximum
per-source flux error falls from 73% to 35%, which is a real improvement, but
remains five times the oracle value. `inner` recovers from `-0.7281` to
`-0.2648` and `lens` from `-0.1938` to `-0.0495`.

**The improvement comes from the catalog, not from the joint fit.** Fitting
the imaging observation alongside the IFU is *worse* than using the imaging
catalog and fitting the IFU alone, 0.4033 against 0.3546. The extra data does
not help while the catalog is wrong.

**With a correct catalog the joint fit does help, modestly.** The oracle joint
arm improves on the oracle IFU arm by 16% in flux error, 0.0600 against
0.0713, and by 14% in morphology error. So a second observation is worth
having once the catalog is right, and is not a remedy for a wrong one.

**Better resolution produces worse supports and spurious sources.** Detection
on the imaging data returns **13 peaks rather than 10**: the deliberately
clumpy morphologies fragment once the PSF stops blending their knots. Mean
support error also degrades from `-2.40` to `-4.00` px, because a narrower PSF
shrinks each source's above-threshold footprint. Support extent is
threshold-dependent, not resolution-dependent. `east`, one of the clumpy
sources, is the clearest casualty: its signed flux error moves from `+0.3162`
to `-0.4033`.

**The residual again fails to rank the arms.** The oracle joint arm has a
slightly worse IFU chi-square than the oracle IFU arm, 0.947556 against
0.944663, while being the better recovery on both truth metrics.

```bash
PYTHONPATH=../lisasep/src python -m \
  benchmarks.run_synthetic_many_source_joint_imaging \
  --max-iter 2000 --output-dir benchmark_artifacts/many_source_joint_imaging
```

### Soft-constraint ablation

Smoothness is tested with a matched AdaProx baseline because the quadratic
proximal penalties are not part of the variable-projection optimizer. On the
recoverable rank-one cube, spectral strength 300 is the selected arm: maximum
spectrum, morphology, total-flux, line-flux, and line-peak errors are 4.0%,
14.3%, 3.2%, 7.3%, and 5.5%, respectively. Its `chi2/N = 0.9553` is deliberately
worse than the unregularized training value 0.9449; selection is based on truth
recovery and line preservation, not minimum training residual. Spectral
strength 3000 suppresses line peaks by as much as 20.7%. Spatial strength 100
is rejected because maximum morphology and integrated-flux errors rise to
22.8% and 10.8%, and the minimum identity margin falls below 0.8.

Those numbers are the 300-iteration values. Re-running every arm at 2,000
iterations leaves the **arm ordering unchanged on all eight metrics**, so the
`spectral_300` selection is not an artifact of the budget. Two of the stated
reasons are, however:

- `spectral_3000`'s line-peak suppression is 7.2% at convergence, not 20.7%,
  and its `chi2/N` falls from 1.0291 to 0.9562. Most of the peak loss was an
  optimization transient. It still loses to `spectral_300` on line peak
  (5.2%), so the choice stands on a much smaller margin than published.
- `spatial_100` gets *worse* with budget on spectrum, flux and identity
  margin (0.1094 to 0.1333, 0.1080 to 0.1302, 0.7805 to 0.7345) while its
  projected gradient *rises* from `8.2e-04` to `1.3e-03`. That arm is not
  converging, so its rejection is sound but its recorded numbers are not a
  converged measurement of the penalty.

`constraint_ablation.json` now also records `max_iter`, the derived
`best_arm_per_metric`, and `selection_wins_metrics`. The `selection` and
`reason` fields are declared constants; previously they were written
unconditionally and could not disagree with the numbers beside them.

```bash
python -m benchmarks.run_synthetic_many_source_constraint_ablation \
  --output-dir benchmark_artifacts/many_source_constraint_ablation_converged \
  --max-iter 2000
```

```bash
python -m benchmarks.run_synthetic_many_source_constraint_ablation \
  --output-dir benchmark_artifacts/many_source_constraint_ablation
python -m pytest -q tests/test_ifu_many_source_constraints.py
```

### Chromatic-morphology recovery contract

The stronger `chromatic_source_cubes` fixture deliberately violates rank one:
each of the same ten galaxies has an offset, clumpy emission-line morphology
in addition to its continuum morphology. A deterministic 1/11 of otherwise
valid voxels is held out during fitting. Even with oracle spectral shapes, one
morphology per galaxy fails (`held-out chi2/N = 2.54`, maximum source-cube
error 65%). The continuum-plus-line geometry reaches held-out `chi2/N = 0.996`;
the start chosen by minimum held-out chi-square has maximum line-flux error
5.4%, line-centroid error 0.16 pixel, and source-cube error 22.6%. A second
correlated-noise draw retains held-out `chi2/N = 1.003` and line-flux error
below 4.5%.

The oracle-spectrum arm is intentionally a geometry/identifiability positive
control, not a proposed science estimator. Giving both components unrestricted
free spectra improves the scene residual but fails source attribution (source
cube error above 60% and integrated-flux error above 30%). The real pipeline
must therefore use independently justified continuum/line spectral structure
or external morphology information before adding chromatic factors; merely
doubling free factors would make the lens-confusion problem worse.

Variable projection now supports source-specific fixed wavelength supports,
profiling only the active sources in each channel. This is exact for positive
tabulated spectra and is tested independently. Minimum-volume regularization
with mixed supports remains rejected because its coupled spectral update does
not yet preserve those zeros.

```bash
python -m pytest -q tests/test_ifu_chromatic_morphology.py
python -m benchmarks.plot_synthetic_chromatic_morphology \
  --output-dir benchmark_artifacts/chromatic_morphology
```

The chromatic plots retain absolute pixel coordinates. Cyan contours denote
continuum truth, lime contours line-map truth, and the source gallery overlays
the corresponding recovered contours.

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

### Grouped-factor diagnostics

The runner accepts `+`-joined catalog names in `--sources`, for example
`--sources 'lens,lz1+L5,lz2,...'`. Such a token fits one rank-one factor over
the smallest odd square containing the standard supports of all members. The
group uses positivity without an exact centroid because the catalog does not
specify the members' relative fluxes. The JSON report records the expanded
`source_groups` mapping and actual combined support. This is a model-selection
diagnostic, not evidence that grouped catalog entries are one physical galaxy.

This distinction matters for the crowded `lz1`/`L5` region. Arribas et al.
classify `lz1` as a foreground galaxy at z=2.576 and `L5` as a z=6.90940 line
emitter. The separate-factor public-cube arm contains the expected [O III]
doublet in the `L5` spectrum, at 3.923 and 3.961 um, but also assigns the same
lines to `lz1`; their fitted spectral cosine is 0.647. On the complete
2.87--5.27 um G395H cube, which retains redshifted H-alpha near 5.19 um,
separate and grouped 30-iteration arms give `chi2/N = 1.231236` and `1.231351`,
respectively. The small likelihood difference is not decisive after accounting
for an entire extra 3,610-channel spectrum. Moreover, the grouped morphology
centroid moves away from both catalog positions, showing that its broad support
is absorbing unrelated structure. The MAST G395H cube therefore supports
localized high-redshift line emission but does not by itself establish the
paper's stronger separate-galaxy interpretation; that classification also uses
shorter-wavelength R100 features absent from G395H. Source-level products keep
`L5` and the three `lz` objects separate while marking their attribution as
unresolved; a merged factor may only be labeled and used as a nuisance
component.

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
