# Kinematic IFU Deblender for JWST — Staged Build Plan

A tool to **jointly deblend a central point source (SN/AGN) from its host AND fit the
host's kinematics, across the full JWST IFU cube**, using scarlet-style **exact proximal
constraints** on a **differentiable forward model**. No neural prior. No nucleus masking.

## Why this project exists (the gap)

**Core idea (lead with this):** a source is *self-similar* — **one SED across all wavelengths, one
morphology across all spaxels**. Existing IFU codes don't use this for deblending: **per-slice**
deblenders fit each wavelength independently (waste color self-similarity); **per-spaxel** spectral
fitters fit each pixel independently (waste morphology self-similarity); kinematic forward-models use
shape consistency but are **single-source and mask the nucleus**. spaxlet brings scarlet's whole-cube
color+morphology self-similarity (constrained matrix factorization) to IFU deblending **for the first
time**, with **exact constraints** (no soft penalties, no trained prior). The IFU amplifies it: ~1000s
of channels make each SED a near-unique fingerprint that ~5 broadband colors blur away.

**The math — "low-rank → low-parameter":** without kinematics each source is a clean rank-1 **outer
product** `S(λ) ⊗ M(x,y)` (linear, cheap — the self-similarity). A velocity field entangles λ and
(x,y): `source = M(x,y)·T(λ; v(x,y), σ(x,y))`, which is **no longer rank-1 / not an outer product**.
We reparametrize the line as one morphology × a shifted/broadened template controlled by scalar maps
(disk → ~6 params) — trading *low-rank* for *low-parameter*. The continuum host and point source stay
clean outer products; only the line, over its narrow window, is nonlinear. **VarPro** keeps the
closed-form linear solve for the amplitude/morphology block and does nonlinear opt only on the
velocity geometry. (This is also *why deferring kinematics is clean:* M1 is the pure outer-product
factorization; kinematics is a small, contained nonlinear perturbation on one component.)

### The competing tools
- **GalPaK³ᴰ / ³ᴰBarolo / qubefit**: fit host kinematics in a cube, but **mask the nucleus** —
  cannot handle a bright central point source.
- **Vietri+ 2024 (arXiv:2411.13270)**: deblend AGN/host **per wavelength slice, independently** —
  **no kinematics**, degrades when AGN isn't dominant.
- **arXiv:2510.27214 (JWST)**: 2D single-band Sérsic+PSF — **no cube, no kinematics**, and
  documents that **Sérsic models are degenerate/inadequate** for AGN hosts.
- **RUBIX (arXiv:2412.08265)**: JAX/GPU IFU forward *simulator* — not a fitter/deblender, but a
  reusable render/PSF/LSF engine.

**Nobody does joint point-source-deblend + kinematic fit across the cube with hard constraints.**
That's the contribution. arXiv:2510.27214's Sérsic-degeneracy complaint is our cited motivation;
exact proximal constraints are our cure for it.

## Core architectural decision (locked)
- **Foundation:** standalone tool. **NOT** scarlet2 (its gradient/soft-penalty + neural-prior
  setup lost scarlet1's exact constraints and was slower). **NOT** scarlet1's autograd+bilinear
  engine (can't host a per-pixel velocity render).
- **Engine = proximal gradient (proxmin) with autodiff'd likelihood gradient.** Constraints stay
  **exact projections** (what worked in scarlet1); the *forward model / gradient* becomes a
  flexible differentiable cube model (adds kinematics, kills rank-1 limitation).
  ```
  θ⁺ = prox_C( θ − η · ∇_θ L(θ) )
       ∇L  : autodiff through point src + PSF⊗LSF + disk + line   (flexible)
       prox: exact non-neg / monotonic / symmetric / TV-smooth     (scarlet1, kept)
  ```
- **Hybrid constraint handling:**
  - free pixel-grid morphology (continuum host) → **proximal projection** (no reparametrization exists)
  - physical params (disk geometry, fluxes, redshift) → **reparametrization** (disk model *is* the constraint; pure autodiff)
- **Why fast:** JWST IFU cubes are spatially **tiny (~30×30 spaxels)**; the expensive per-pixel
  Doppler render runs only on **line-window channels (dozens)**, not the full spectral axis.
- **Joint-fit gradient mechanism = implicit differentiation (optimization layers, JAXopt).** The
  deblend and kinematic fit are nested optimizations. Differentiate through the **solution** (implicit
  function theorem) rather than **unrolling** the inner solver → exact gradients, constant memory,
  fast. *This is the intended mechanism from Stage 1/2, and the likely cure for scarlet2's slowness
  (which came from gradient/unrolling-style fitting of soft penalties).*
- **Landscape guardrail:** the SED×morphology model is a structured **low-rank factorization** →
  non-convex. Recent benign-landscape theory says shallow/low-rank factorizations have no spurious
  minima (strict saddles) given enough data. **Design rule:** keep components few / rank low, and
  use **two-stage init** (spectral/tailored init → proximal refine) to land in the global basin.
- **Scaling lever (later):** variance-reduced **stochastic** proximal gradient lets us **minibatch
  the cube** (subsample channels/spaxels per step) for speed *while keeping exact constraints*, with
  convergence guarantees. Adopt in Stage 3/5 if needed, not in the MWV.

## Reusable parts (don't rebuild)
| Need | Source | In-repo location |
|---|---|---|
| Proximal operators (monotonic, symmetry, positivity, mask) | scarlet1 / proxmin | `scarlet/operator.py`, `scarlet/constraint.py` |
| FFT convolution / PSF matching | scarlet1 | `scarlet/fft.py` (`convolve`, `match_psf`) |
| PSF models | scarlet1 | `scarlet/psf.py` (Gaussian/Moffat/ImagePSF) |
| Disk kinematic parameterization (incl, PA, arctan rot curve, σ) | GalPaK³ᴰ | external ref |
| JAX render + PSF⊗LSF 3D-kernel convolution | RUBIX (open source) | external, adapt |
| Per-channel JWST PSF cube | WebbPSF | external |
| Optimizer / posteriors | optax / numpyro / jaxopt | external |

---

# STAGES

## Stage 0 — Skeleton & synthetic data generator  *(foundation; ~days)*
**Goal:** a repo that can *make* a fake cube with known truth. Nothing fitted yet.
- New package (own repo or `scarlet/kinematic_ifu/`), JAX + proxmin deps.
- **Synthetic cube generator** (the truth machine): rotating-disk emission-line host
  (params: incl, PA, center, v_max, r_t, σ, line flux, line λ₀/z) + a smooth continuum
  + a PSF point source (flux, position, point-source spectrum) ⊗ per-channel PSF + noise.
- Tiny grid: 30×30 × (one line window, ~50 channels). Gaussian noise to start.
- **Exit test:** generate a cube, eyeball channel maps show a rotating line + a point source.
- *No fitting. This is the ruler you measure everything against.*

## Stage 1 — Minimal Working Version (MWV)  *(the milestone — get it RUNNING)*
**Goal:** recover injected parameters from a synthetic cube. Simplest possible everything.
- **Forward model (JAX, differentiable):**
  - point source: `flux · PSF_λ` at fitted (x₀,y₀), with a free point-source spectrum
  - host: **single parametric rotating-disk emission line** (incl, PA, center, v_max, r_t, σ, flux)
    — *no free morphology yet; disk is fully parametric* = pure reparametrized autodiff, no prox needed
  - render = sum, ⊗ PSF per channel (reuse fft.convolve concept), + (optional) LSF
- **Fit:** plain autodiff gradient descent (optax Adam) — *no proximal step yet*, because
  everything is reparametrized (fluxes via softplus, angles bounded). Get the loop working first.
  *(If the inner kinematic solve is ever nested, use JAXopt implicit diff — don't unroll.)*
- **Exit test (identifiability):** inject known disk+point source, recover params within tolerance.
  Critically: **can it separate point-source flux from nuclear rotation?** If not, learn it cheaply here.
- **Deliverable:** one script, synthetic-in → recovered-params-out, a residual plot. *This is the MWV.*

## Stage 2 — Add the proximal engine + free continuum host  *(the scarlet1 soul)*
**Goal:** introduce exact proximal constraints — the thing that worked in scarlet1.
- Add a **free pixel-grid continuum host** component (not parametric) with **proximal constraints**:
  non-negativity + monotonicity + symmetry (reuse `scarlet/operator.py` prox ops, ported to
  work on JAX arrays — they're simple numpy ops).
- Switch outer loop to **proximal gradient**: `θ⁺ = prox_C(θ − η∇L)`, with ∇L from autodiff and
  prox applied to the free-morphology block only (disk/fluxes stay reparametrized).
- **Exit test:** recover a host with both smooth continuum (constrained morphology) AND a
  rotating line (parametric), with a point source on top. Constraints **hold exactly** (verify
  monotonicity/positivity are satisfied, not just penalized) and it's **faster** than soft-penalty.
- *This is where you prove the architecture beats the scarlet2 experience: exact + fast.*

## Stage 3 — Realism: JWST PSF, LSF, correlated noise  *(make it real)*
- **WebbPSF per-channel PSF cube** for NIRSpec IFU / MIRI MRS (replace toy PSF).
- **LSF** convolution along spectral axis.
- **Correlated-noise-aware likelihood**: at minimum down-weight via a proper covariance/weight
  handling so error bars aren't overconfident (the cube_build covariance problem).
- Handle **undersampling** at NIRSpec blue end (sub-spaxel PSF).
- **Exit test:** run on a *realistic* synthetic cube (WebbPSF + correlated noise) and still recover.

## Stage 4 — Validation against the field  *(credibility)*
- Reproduce **GalPaK³ᴰ** on a masked-nucleus disk (no point source) → must match it.
- Reproduce **Vietri+ 2024 (2411.13270)** per-slice deblend on MaNGA-like → show joint+kinematic
  does better, esp. when AGN is *not* dominant (their stated weakness).
- Show recovery of **nuclear kinematics that masking throws away** — the headline result.
- Run on **1–2 real JWST IFU cubes** (a SN-in-host and/or an AGN-host).

## Stage 5 — Hardening into the reusable pipeline tool  *(the deliverable)*
- I/O: read `jwst`-pipeline `s3d` cubes + error/DQ arrays directly.
- Config-driven (which components, which constraints per component, line list/redshift prior).
- Posteriors: swap optax → **numpyro/NUTS** for uncertainties on disk params (optional mode).
- Docs, tests, examples, pip-installable. Benchmark numbers vs. GalPaK³ᴰ / 2411.13270.

---

# Modularity contract (so improvements don't force rewrites)
Keep these **swappable behind interfaces** from Stage 1 on:
1. **Components** (point source / parametric disk / free morphology / line template) —
   each is a `render(params) → cube_contribution`. Add component types without touching the loop.
2. **PSF/LSF model** — toy ↔ WebbPSF behind one `psf(λ) → kernel` interface.
3. **Constraints** — each component declares its prox set (or "reparametrized, none").
4. **Likelihood / noise model** — diagonal ↔ correlated behind one `loss(model, data, weights)`.
5. **Optimizer** — optax Adam ↔ proximal-gradient ↔ numpyro behind one `fit(model, data)`.

# Decision log / open forks (revisit at the marked stage)
- **Parametric disk vs. free velocity map** → start **parametric** (Stage 1, fast/robust);
  add optional **free-residual velocity correction** only if a target needs it (post-Stage 4).
- **Joint vs. staged fit** → start **joint** (Stage 1); it's why we exist (nucleus recovery).
- **Build own repo vs. live in scarlet1** → lean **own repo**, reuse scarlet1 ops as a dependency.
- **scarlet2 reuse** → **no** (lost constraints, slower, neural prior).

# First concrete actions (Stage 0 → 1)
1. Create package skeleton + deps (jax, optax, proxmin, astropy, webbpsf later).
2. Write `synth.py`: rotating-disk-line + continuum + point-source + PSF + noise → cube + truth.
3. Write `forward.py`: differentiable render of {point source, parametric disk line}.
4. Write `fit_mwv.py`: optax Adam loop; recover injected params; residual + truth-vs-fit plot.
5. **Identifiability test** = the go/no-go for the whole project.
