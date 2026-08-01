# lisasep IFU parity harness

The fixtures in `ifu_parity_contracts.py` pin the 17x19 operator cases and the
six-channel end-to-end feature matrix before production behavior is changed.
They include a compatible null, a close smooth blend, and a deliberately
clumpy negative control.

Run Scarlet's local contract tests:

```bash
python -m unittest discover -s tests -p 'test_ifu_parity_contracts.py'
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
