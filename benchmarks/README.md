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
budget as the corrected lisasep comparison. Submit all predeclared starts with:

```bash
sbatch submit_collaborator_reproduction.sh
```

For a predeclared time-to-equal-fit comparison with a larger iteration cap,
keep the original products isolated by setting both batch variables:

```bash
sbatch --export=ALL,SCARLET_MAX_ITER=1500,SCARLET_RUN_LABEL=matched1500 \
  submit_collaborator_reproduction.sh
```
