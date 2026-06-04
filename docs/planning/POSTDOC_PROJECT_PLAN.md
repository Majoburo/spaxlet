# Project Plan — Whole-Cube Self-Similar Deblending of JWST IFU Cubes
### (color + morphology self-similarity; point-source/host separation, kinematics later)

**For:** Kirsty Taggart
**PI/supervisor:** Maria Jose Bustamante Rosell
**Horizon:** 2 years · **Year 1:** method paper + released tool · **Year 2:** science application + extensions
**Your profile (assumed):** strong on IFU/AGN/kinematics science; ramping up on JAX / autodiff /
proximal optimization. The reading list and early milestones reflect that.

---

## 1. The one-paragraph pitch

A source is **self-similar**: it has *one* spectral energy distribution across all wavelengths and
*one* morphology across all spaxels. Existing IFU codes do not exploit this for deblending —
**per-slice** deblenders fit each wavelength independently (discarding color self-similarity),
**per-spaxel** spectral fitters fit each pixel independently (discarding morphology self-similarity),
and kinematic forward-models (GalPaK³ᴰ, ³ᴰBarolo) use spatial consistency but assume a *single*
source and **mask the bright nucleus** — the region of greatest interest. We bring scarlet's
whole-cube **color-and-morphology self-similarity** (constrained matrix factorization) to IFU
deblending **for the first time**, with **exact physical constraints** (not soft penalties, not a
trained prior). An IFU makes it far more powerful than broadband: ~1000s of channels turn each
source's SED into a near-unique *fingerprint* that ~5 broadband colors blur away, so overlapping
sources become well-posed to separate. Because the model is self-consistent across the whole cube it
can **hold the nucleus in place instead of masking it**, and it **extends to kinematics** by letting
the otherwise-consistent emission line be shifted by a velocity field — recovering the nuclear
rotation that masking-based tools discard. SN/AGN-in-host is the *application*; the *method gap* is
the headline. The engine combines a **differentiable (autodiff) forward model** of the cube with
**scarlet-style exact proximal constraints**, which break the morphological degeneracies that
limit current soft-fitting methods (a documented failure mode of Sérsic decompositions).

## 2. Why it's novel (the gap — verified against the literature)

| Existing tool | Uses color self-similarity? | Uses shape self-similarity? | Deblends point source? | Kinematics? | Exact constraints? |
|---|---|---|---|---|---|
| GalPaK³ᴰ / ³ᴰBarolo / qubefit | ✅ | ✅ | ❌ (masks nucleus) | ✅ disk | ❌ (MCMC) |
| Vietri+ 2024 (2411.13270) | ❌ (per-slice) | ✅ | ✅ | ❌ | ❌ |
| Li+ 2025 JWST (2510.27214) | ❌ (1 band) | ✅ | ✅ | ❌ | ❌ |
| per-spaxel fitters (pPXF-style) | ✅ | ❌ (per-spaxel) | ❌ | ✅ | ❌ |
| **This project (spaxlet)** | ✅ | ✅ | ✅ | ✅ disk | ✅ proximal |

**No existing IFU code uses BOTH color and morphology self-similarity to deblend across the whole
cube with exact constraints.** That joint, whole-cube self-similarity is our contribution — the
deblend is the foundation, kinematics is the extension.

---

## 3. Reading list (in order)

### Tier A — read first (the project's foundations)
1. **Melchior et al. 2018**, *scarlet* (arXiv:1802.10157 / Astron. Comput. 24, 129) — the
   constrained matrix factorization, proximal constraints, and the deblending philosophy we build on.
   *Focus: §on constraints (monotonicity, symmetry, positivity) and the proximal gradient method.*
2. **Bouché et al. 2015**, *GalPaK³ᴰ* (arXiv:1501.06586) — the parametric disk kinematic forward
   model we adopt (inclination, PA, arctan rotation curve, σ; 3D PSF⊗LSF kernel). *This is half our model.*
3. **Vietri et al. 2024** (arXiv:2411.13270) — per-slice AGN–host IFS deblending. *Our closest
   competitor; understand exactly why per-slice + no-kinematics is the thing we improve on.*
4. **Li et al. 2025** (arXiv:2510.27214) — JWST AGN–host decomposition & **Sérsic limitations**.
   *Our cited motivation: documents the degeneracies our hard constraints address.*

### Tier B — technical ramp-up (the JAX/optimization side — your growth area)
5. **JAX docs**: "JAX 101" + Autodiff Cookbook + `jit`/`vmap`/`grad`. *Hands-on, not just reading.*
6. **optax** quickstart (Adam etc.) and **numpyro** intro (for later Bayesian/NUTS posteriors).
7. **Proximal gradient methods** — a short tutorial (e.g. Parikh & Boyd 2014, *Proximal Algorithms*,
   §1–3). *Just enough to understand `θ⁺ = prox_C(θ − η∇L)` and what a projection does.*
8. **proxmin** (Melchior et al.) README/docs — the proximal library scarlet uses, which we reuse.
9. Skim **RUBIX** (arXiv:2412.08265) + its GitHub — open-source JAX IFU render; we borrow PSF⊗LSF code.

### Tier B+ — relevant optimization theory (recent; "may make the engine faster/cleaner", not required up front)
*These are techniques and results that directly bear on our engine. Skim for ideas, not mastery.
The first item is the most important — it's our intended mechanism for the joint fit.*
- **Implicit differentiation of optimization layers** — JAXopt implicit-diff + deep-equilibrium
  tutorial (https://jaxopt.github.io ; http://implicit-layers-tutorial.org). *Differentiate
  through the **solution** of an inner optimization via the implicit function theorem, instead of
  unrolling every iteration → exact gradients, constant memory, much faster. This is how we couple
  the deblend and the kinematic fit. Plausibly fixes the slowness that hurt scarlet2.*
- **∇-Prox: Differentiable Proximal Algorithm Modeling** (Princeton, deltaprox) — a framework that
  is essentially our engine (proximal + autodiff) generalized. *Evaluate as a reference/foundation.*
- **Variance-reduced stochastic proximal gradient** (e.g. arXiv:2401.12508) — *principled way to
  **minibatch the cube** (subsample channels/spaxels per step) for speed while keeping exact
  constraints, with convergence guarantees. Adopt when we need to scale.*
- **Benign-landscape theory for low-rank/bilinear factorization** (Chen et al. overview; 2023–26
  deep-factorization results) — *our SED×morphology model is a structured low-rank factorization.
  Justifies our two-stage (spectral init → refine) strategy and the design rule "keep the
  factorization shallow (few components, low rank)" to avoid spurious minima. Cite in the paper.*
- **Learned Proximal Networks** (ICLR 2024, arXiv:2310.14344) — *not something we use (no learned
  prior), but the theory of "what makes a valid proximal operator" is exactly the property
  scarlet2's soft penalties lacked. Read as the argument for why our exact-projection choice is
  sound, and as an optional future door (a data-driven prior that keeps convergence guarantees).*

### Tier C — JWST instrument reality (read by Stage 3)
10. **NIRSpec IFU** docs (JWST User Docs) + **Böker et al. 2022** (NIRSpec IFU, arXiv:2202.xxxx).
11. **MIRI MRS** docs + **Law et al. 2023** 3D-drizzle (arXiv:2306.05520) — cube build & correlated noise.
12. **WebbPSF** docs — per-channel PSF model generation.

### Tier D — context / kinematics validation
13. **³ᴰBarolo** (Di Teodoro & Fraternali 2015) and a recent high-z kinematics-tool comparison
    (e.g. arXiv:2509.18328) — how the community validates kinematic recovery; our benchmarks.

> **Suggested first two weeks:** Tier A (1–4) for the *why*, then Tier B (5–7) hands-on for the
> *how*. Don't try to read everything before coding — start Stage 0 by week 2.

---

## 4. Architecture (the decisions already made — context, not for re-litigation early on)

- **Engine = proximal gradient with autodiff'd likelihood.** Constraints stay **exact projections**
  (scarlet1's strength); the forward model is **differentiable** (adds kinematics).
  `θ⁺ = prox_C( θ − η·∇_θ L(θ) )`.
- **Hybrid:** parametric pieces (disk, fluxes) are **reparametrized** → plain autodiff; free
  pixel-grid morphology (continuum host) uses **proximal projection**.
- **NOT scarlet2** (its soft-penalty + neural-prior setup lost the exact constraints and was slower).
  **NOT** a neural/trained prior — all priors are analytic/physical and inspectable.
- **Fast because** JWST IFU cubes are spatially tiny (~30×30) and the expensive per-pixel Doppler
  render runs only on the narrow emission-line channel windows.
- **Reuse:** proximal ops + FFT convolution + PSF from this scarlet1 repo (`scarlet/operator.py`,
  `constraint.py`, `fft.py`, `psf.py`); disk model from GalPaK³ᴰ; render bits from RUBIX; WebbPSF.
- **Joint-fit mechanism = implicit differentiation (optimization layers, JAXopt).** The deblend and
  the kinematic fit are nested optimizations; we differentiate through the *solution* (implicit
  function theorem), **not** by unrolling the inner solver. Exact gradients, constant memory, fast —
  this is the modern fix for the slowness that hurt scarlet2. (See Tier B+ reading.)
- **Stay in the benign landscape:** keep the factorization **shallow / low-rank** (few components)
  and use **two-stage init** (spectral/tailored init → refine) so we land in the global basin —
  justified by recent benign-landscape theory, not just folklore.

---

## 5. Timeline (24 months, quarter by quarter)

### YEAR 1 — Method + Tool

**Q1 (Mo 1–3): Onboarding + Minimal Working Version**
- *Mo 1:* Tier A+B reading (hands-on JAX). Set up env (jax, optax, proxmin). Reproduce a tiny JAX
  autodiff example end-to-end. Run scarlet1 on its quickstart to feel the constraints.
- *Mo 2:* **Stage 0** — synthetic cube generator (rotating-disk line + continuum + point source +
  toy PSF + noise; 30×30×~50). This is the "truth machine."
- *Mo 3:* **Stage 1 — MWV.** Differentiable forward model (point source + parametric disk line),
  optax Adam fit. **★ Milestone M1 (go/no-go): identifiability test** — recover injected params;
  *can the joint fit separate point-source flux from nuclear rotation?*
- **Deliverable Q1:** running MWV + short internal memo on identifiability (the go/no-go result).

**Q2 (Mo 4–6): The proximal engine + realism begins**
- **Stage 2** — add free continuum-host morphology with **exact proximal constraints**
  (non-neg/monotonic/symmetry, ported from scarlet1); switch to proximal-gradient outer loop.
  *Verify constraints hold exactly AND it's faster than a soft-penalty baseline.*
- **Stage 3a** — swap toy PSF for **WebbPSF** per-channel; add **LSF**.
- **Deliverable Q2:** model that jointly fits {point source + constrained continuum + kinematic
  line} on realistic-PSF synthetic cubes. ★ **Milestone M2.**

**Q3 (Mo 7–9): Realistic noise + validation**
- **Stage 3b** — correlated-noise-aware likelihood / weighting; undersampling at NIRSpec blue end.
- **Stage 4a** — validation: reproduce **GalPaK³ᴰ** on masked-nucleus disks (must match); beat
  **Vietri+2024** per-slice when AGN is *not* dominant; demonstrate **nuclear-kinematics recovery**
  that masking discards (the headline figure).
- **Deliverable Q3:** validation report + headline figures. ★ **Milestone M3 (method validated).**

**Q4 (Mo 10–12): First real data + method paper**
- Run on **1–2 real JWST IFU cubes** (one SN-in-host and/or one AGN-host).
- **Stage 5a** — package: `s3d` I/O, config-driven, basic docs/tests, pip-installable (v0.1).
- **Write the method paper.**
- **Deliverable Q4:** ★ **Milestone M4 — method paper submitted + tool v0.1 released.**

### YEAR 2 — Science + Extensions

**Q5–Q6 (Mo 13–18): Science application**
- Assemble a target sample (AGN hosts / SN hosts with JWST IFU). Run the tool at scale.
- **Stage 5b** — Bayesian mode (numpyro/NUTS) for posteriors on disk params; robustness/uncertainty.
- **Science paper drafting** on nuclear kinematics / host properties of the sample.
- ★ **Milestone M5 — science sample analyzed.**

**Q7–Q8 (Mo 19–24): Extensions + second paper**
- **Free-velocity-residual extension** (optional per-pixel correction on top of the disk) for
  disturbed/merging systems; MIRI MRS support (longer λ, different PSF/sampling regime).
- Tool v1.0 (docs, tutorials, community-ready). Second (science) paper submitted.
- ★ **Milestone M6 — tool v1.0 + science paper.**

> Quarters are targets, not contracts. M1 (identifiability) is the real gate: if it fails, we
> re-scope early and cheaply rather than pushing on.

---

## 6. Milestones at a glance

| ID | When | Milestone | Success criterion |
|---|---|---|---|
| **M1** | Mo 3 | Identifiability (go/no-go) | Recover injected disk+point-source params from synthetic cube |
| **M2** | Mo 6 | Proximal engine + WebbPSF | Exact constraints hold; faster than soft baseline; realistic PSF |
| **M3** | Mo 9 | Method validated | Matches GalPaK³ᴰ; beats per-slice; recovers nuclear kinematics |
| **M4** | Mo 12 | Method paper + tool v0.1 | Paper submitted, code public |
| **M5** | Mo 18 | Science sample | Posteriors on a real target sample |
| **M6** | Mo 24 | Tool v1.0 + science paper | v1.0 released, second paper submitted |

## 7. Deliverables
- **Code:** public, pip-installable JAX package (proximal + autodiff forward-model deblender),
  reusing scarlet1 proximal operators. Docs, tests, tutorials.
- **Papers:** (1) method paper [Yr1], (2) science application [Yr2].
- **Reproducible benchmarks** vs. GalPaK³ᴰ and Vietri+2024.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| **Joint fit not identifiable** (point source vs. nuclear rotation degenerate) | M1 tests this *first*, cheaply, on synthetic data — go/no-go before any realism investment |
| Correlated JWST noise breaks the likelihood | Stage 3b explicitly; down-weight / covariance handling; validate error bars on synthetics |
| Disk model too rigid for real disturbed hosts | Start parametric (robust); free-velocity residual is a Year-2 extension, not a blocker |
| JAX/optimization learning curve | Tier-B hands-on ramp in Mo 1; reuse RUBIX/scarlet code; PI pairing in early stages |
| **Unrolling the inner fit makes it slow/memory-heavy** (the scarlet2 failure mode) | Use **implicit differentiation** (JAXopt optimization layers) — differentiate the solution, not the iterations (Tier B+) |
| **Non-convexity → bad local optimum** | Keep factorization shallow/low-rank + two-stage spectral init; benign-landscape theory says this lands in the global basin |
| scarlet1 prox ops need porting to JAX | They're simple array ops; budget time in Stage 2; fall back to proxmin directly |

## 9. Working rhythm (suggested)
- **Weekly** 1:1 check-in (more frequent during Mo 1–3 onboarding).
- All work in a git repo from day 1; synthetic-data tests are the regression suite.
- Each stage has an **exit test** — don't advance until it passes.
- Keep a running decision log (the architecture choices are in `KINEMATIC_IFU_PLAN.md`).

## 10. Companion documents
- **`KINEMATIC_IFU_PLAN.md`** — the detailed technical build plan (stages, exit tests, modularity
  contract, reusable-parts table). Read alongside this.
