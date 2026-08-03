# lisasep IFU additions to Scarlet

This branch is a surgical IFU compatibility and validation layer, not a
rewrite of Scarlet. Upstream imaging behavior remains the default; the new IFU
renderer, morphology constraints, variable-projection optimizer, and volume
term are opt-in.

## Reproducibility anchors

- Branch: `lisasep-ifu-parity`
- Upstream base: `3ce064d714d27f8dcbdb9a77c438272960697d16`
- Implementation tip before this note: `3dd5153`
- Comparison environment: Scarlet `1.0.1+g3ce064d`
- lisasep reference: `8dbafc835fc5712bac10c882810b05a1bdc456de`
- Full ordered history: `git log --reverse --oneline master..lisasep-ifu-parity`

## What changed and where to look

| Area | Addition or fix | Primary files | Commits |
|---|---|---|---|
| Matched forward model | Explicit intrinsic-frame `DeltaPSF`; corrected PSF matching and factor initialization | `scarlet/frame.py`, `scarlet/psf.py`, `scarlet/observation.py`, `scarlet/renderer.py`, `scarlet/blend.py` | `f439243`, `85a76d1`, `8b13357`, `24255dd` |
| IFU ingestion | Mask-safe `Observation.from_ifu_arrays`, measured variance, physical wavelength grids, strict channel compatibility | `scarlet/observation.py`, `scarlet/frame.py` | `4979b2e`, `e13b915`, `ed95b6e` |
| IFU PSFs | Channel-mapped PSFs plus deterministic crop, centroid measurement, and subpixel recentering | `scarlet/ifu.py`, `scarlet/renderer.py`, `scarlet/fft.py` | `5dbff71`, `bc85078` |
| Spatially varying PSF | Opt-in fixed field-PSF renderer with bilinear spatial interpolation and adjoint | `scarlet/ifu.py`, `scarlet/renderer.py` | `538b457` |
| Constraints | Coordinate-aware constraints; centered symmetry/monotonicity arms; exact Dykstra intersection of positivity and a linear centroid constraint; resize-safe centers | `scarlet/constraint.py`, `scarlet/morphology.py` | `a62d79a`, `d6a1ef7`, `d1c68e3`, `1307683`, `bbd25b5`, `96f23ee`, `96d9d11`, `1e5cd16` |
| Optimization | Bounded-memory channel chunks, projected-gradient/KKT diagnostics, KKT stopping, and scalar-metric constrained variable projection | `scarlet/blend.py`, `scarlet/optimization.py`, `scarlet/renderer.py` | `9de4e8c`, `8557251`, `e6b5a06`, `02e1ed2`, `ac43f63` |
| Identifiability | Exact pairwise non-negative mixing intervals/envelopes and an active normalized spectral log-volume objective | `scarlet/degeneracy.py`, `scarlet/optimization.py` | `dc65f95`, `ac43f63`, `e52ccb2` |
| PSF coordinate frame | Centroid constraints and truth scores now use the latent frame implied by PSF recentering | `benchmarks/run_collaborator_reproduction.py`, `benchmarks/ifu_parity_metrics.py` | `1633662`, `3029d51` |
| Reproduction/plots | Shared contracts and strict metrics, provenance, batch driver, notebook, latent/rendered factor plots, and converged spectral offsets | `benchmarks/`, `submit_collaborator_reproduction.sh` | `6e7c9aa`, `61f539d`, `9306805`, `be8614e`, `712bc90`, `3dd5153` |

The most useful API entry points are:

```python
observation = scarlet.Observation.from_ifu_arrays(...)
kernels, retained = scarlet.crop_psf_kernels(kernels, size)
kernels, removed_shift = scarlet.recenter_psf_kernels(kernels)

iterations, objective = blend.fit(
    max_iter,
    optimizer="variable_projection",
    channel_chunk_size=64,
    minimum_volume_strength=0,  # optional; keep zero unless validated
)
diagnostics = blend.parameter_optimization_diagnostics()
```

The matched command-line configuration is:

```text
--optimizer variable_projection --feature centroid_psf
```

The old `adaprox` optimizer and the ordinary positivity feature remain
available and unchanged as defaults.

## Important interpretation

PSF recentering removes a median `(dy, dx) = (0.5241, 0.5197)` pixel shift.
The fitted intrinsic morphologies therefore live in a correspondingly shifted
latent frame. Do not compare them directly with catalog-frame truth. Register
the fitted morphology by the negative offset, or compare the PSF-rendered
sources. Commits `3029d51` and `3dd5153` implement the corrected scoring and
plots.

For converged start A, the corrected diagnostics are:

- spectrum relative-L2: `0.87% / 1.10%`;
- registered intrinsic morphology relative-L2: `1.63% / 2.11%`;
- rendered morphology relative-L2: `0.95% / 1.30%`;
- observable cube chi-square/voxel: `0.95480`, essentially the same residual
  floor as lisasep.

Starts A and B converge to this good observable basin with KKT below `1e-4`.
Start C converges to a poor stationary basin (`chi-square/voxel ~= 1.563`).
The volume term is mathematically active and cheap, but strengths `100` and
`1000` did not rescue C. It is experimental, not a recommended default.

## Reading and validation order

1. `benchmarks/run_collaborator_reproduction.py` for the complete experiment.
2. `scarlet/optimization.py` for variable projection and spectral volume.
3. `scarlet/constraint.py` for exact centroid/positivity projection.
4. `scarlet/observation.py`, `scarlet/ifu.py`, and `scarlet/renderer.py` for the
   IFU forward model.
5. `tests/test_variable_projection.py`, `tests/test_ifu_exact_projection.py`,
   `tests/test_ifu_observation_arrays.py`, and `tests/test_ifu_varying_renderer.py`
   for focused contracts.

The focused suite has 50 passing `test_ifu*.py` tests plus four passing
variable-projection tests. Truth-referenced metrics are diagnostics for
declared mocks only; real-data run selection must use truth-independent
residual and optimality criteria.
