# START HERE

Welcome. This is the 5-minute orientation. The detailed docs are in `docs/planning/`,
but read **this** first.

## What we're building (one sentence)

**spaxlet** uses a source's **color + shape self-similarity** to separate a bright point source
(a supernova or AGN) from its host galaxy in a JWST IFU data cube — and, later, also fits the
host's rotation.

## Why it's worth doing (the method gap)

A source is **self-similar**: it has *one* spectrum (color) across all wavelengths and *one*
shape (morphology) across all spaxels. The existing IFU codes don't use this for deblending:
- **per-slice** deblenders fit each wavelength on its own → waste the *color* self-similarity;
- **per-spaxel** spectral fitters fit each pixel on its own → waste the *shape* self-similarity;
- **kinematics codes** use the shape, but only for *one* source, and **mask the bright nucleus**.

We're the **first to bring whole-cube color+shape self-similarity to IFU deblending** (this is
scarlet's idea, never applied to IFUs). An IFU makes it powerful: ~1000s of channels turn each
source's spectrum into a near-unique *fingerprint* that ~5 broadband colors blur away. Because the
model is consistent across the whole cube, we can **keep the nucleus instead of masking it** — and
later add rotation by letting the (otherwise consistent) line shift with a velocity field.
SN/AGN-in-host is the *application*; the *method gap* is the headline. (Papers to cite: `docs/planning/`.)

## The core idea (what makes us different)

We model the cube as a **sum of components** — `point source + host` — where each component is
**self-similar**: one spectrum × one shape (a clean "outer product"). We fit them by
**proximal gradient**:

```
  θ⁺ = prox_C( θ − η · ∇L(θ) )
        ∇L : gradient of the fit, from autodiff (JAX) through the whole model
        prox: snap the result onto hard constraints (positivity, monotonicity, symmetry)
```

The **`prox` step enforces the constraints EXACTLY**, every iteration — it's a projection,
not a penalty. This is the one thing that must not change. (scarlet2 relaxed these into
soft penalties + a trained prior; it lost the constraints and got slow. We do NOT do that.
We have **no trained/neural prior** — every prior is physical and inspectable.)

## Why it's fast

JWST IFU cubes are **spatially tiny** (NIRSpec IFU is literally 30×30 spaxels). The cube is
big only along wavelength, and the expensive parts run only on the few channels that matter.
On a GPU with JAX this is seconds, not hours.

---

## Your first month — do these, in order

We deliberately **start without kinematics** (no rotation yet) to prove the engine first.

1. **#1** Scaffold the package (JAX + proxmin + optax). _Some plumbing already exists — see below._
2. **#3** Synthetic "truth machine": make a fake cube = point source + host (continuum + a
   *static* emission line) + PSF + noise, with known answers.
3. **#5** Differentiable forward model that renders that scene (JAX).
4. **#6** Fit it with Adam; recover the injected answers.
5. **#7** ⭐ **THE GATE:** can the fit cleanly separate the point source from the host, with the
   **exact constraints holding**, **faster than a soft-penalty version**?

### #7 is the whole point of month one

If #7 works, the rest is mostly engineering and we keep going. If it doesn't, we stop and
re-think — cheaply, at month 3, instead of month 12. Treat #7 as **the priority**, not "step 5."

(The soft-penalty version in #7/#13 is a **throwaway control** we build only to show ours is
better. It is *not* our method.)

---

## What already exists (don't rewrite it)

There's prior code in this repo from an earlier attempt — reuse it:
- `junk/spaxlet.py::getdata()` — reads a cube + error → weights, builds the wavelength axis,
  cleans NaNs, masks bad wavelength bins. **This is most of issue #21 (I/O), already done.**
- `junk/wavelength_to_rgb.py` + the point-pickers — cube visualization & source selection (**#4**).
- `scarlet/operator.py`, `scarlet/constraint.py` — the exact proximal operators to port to JAX (**#9**).

Each relevant issue has a comment pointing at the code to reuse.

---

## After the engine works (later milestones — don't worry about these yet)

- **M2:** exact-constraint proximal engine on a free-morphology host + real JWST PSF (WebbPSF).
  Then **add the rotating-disk kinematics** as one new component (#35) — it slots in, no rewrite.
- **M3:** validate (match GalPaK³ᴰ, beat per-slice deblenders) and the headline result:
  **recover the nuclear rotation that masking throws away** (#20).
- **M4:** method paper + `pip install spaxlet` v0.1.
- **Year 2:** run a real target sample, posteriors, MIRI, second paper.

The board shows all of this on a timeline: see the **Project** and **Milestones** tabs.

---

## The few "clever tricks" we care about (so they don't get lost)

You'll see these in issue comments where they're used — they're the reason this is fast/correct:
- **Exact prox over an autodiff gradient** (the core engine — #11).
- **Implicit differentiation** for any nested fit — differentiate the *solution*, don't unroll
  the solver (the fix for scarlet2's slowness — #16).
- **Variable Projection (VarPro)** — solve the linear amplitudes in closed form, only optimize
  the nonlinear geometry (#33).
- **Line-window render** — only render the few channels around an emission line (#5).
- **Parametric disk** for kinematics (~6 numbers), not a free per-pixel velocity map (#35).

You don't need all of these for month one. #1→#7 only needs the first idea.

---

## How we'll work

- Weekly check-in (more often during the first month).
- Everything in git; the synthetic "truth machine" (#3) is our test suite.
- **Q1 is concrete; later quarters are a sketch — push back and make the project yours.**

Questions? The narrative version is `docs/planning/POSTDOC_PROJECT_PLAN.md`; the technical build
plan is `docs/planning/KINEMATIC_IFU_PLAN.md`.
