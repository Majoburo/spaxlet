# lisasep IFU parity branch

This branch is a surgical compatibility and validation effort, not a rewrite
of Scarlet.  Upstream behavior remains the default; new morphology or IFU
features must be explicit and independently gated.

## Reproducibility anchors

- Scarlet remote: `https://github.com/pmelchior/scarlet.git`
- Scarlet base: `3ce064d714d27f8dcbdb9a77c438272960697d16`
- Installed comparison version: `1.0.1+g3ce064d`
- lisasep reference: `8dbafc835fc5712bac10c882810b05a1bdc456de`
- lisasep branch at handoff: `ifu-lowrank-pipeline`

The comparison environment was installed directly from the Scarlet Git
commit above, not from an editable source checkout.  This source checkout was
created at that exact commit before the feature branch was made.

## Promotion gates

Every ported feature must pass:

1. exact operator parity where the two codes intend the same projection;
2. each optimizer's own convergence or optimality diagnostic;
3. strict scale-sensitive recovery or held-out prediction on a predeclared
   compatible mock;
4. the deliberately clumpy misspecification negative control;
5. recorded runtime, memory, and start sensitivity; and
6. truth-independent model selection.

Truth-referenced metrics diagnose a declared experiment.  They must never be
used to select a run presented as applicable to real data.
