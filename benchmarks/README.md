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
  --python /path/to/venv-scarlet/bin/python
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
Saved metrics and NPZ products also contain exact two-sided pairwise mixing
intervals plus integrated-spectrum and unit-flux-morphology envelopes. These
are labeled structural sensitivity floors, not posterior or `+/-1 sigma`
uncertainties.
