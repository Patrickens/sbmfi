# sbmfi — PhD Project Overhaul Plan

> **Status legend:** `[ ]` = todo · `[~]` = in progress · `[x]` = done · `[!]` = blocked

Last updated: 2026-04-14

---

## Context

`sbmfi` (Simulation-Based Metabolic Flux Inference) is a PhD project implementing ¹³C metabolic flux analysis via simulation-based inference. The core is built around cobrapy models extended with atom-mapping, EMU simulation, and a polytope-based flux prior. The project needs to be brought to a professional, publishable state before job applications and a final paper.

**Execution order:**

1. **Phase 1 — Delete LinAlg** — `LinAlg('numpy' | 'torch')` wrapper is dead weight; go torch-only with direct calls.
2. **Phase 2 — MCMC / ABC refactor** — Current implementation is buggy/broken/slow. Rewire `MCMC` and `SMC` to use the correct `MarkovTransition` kernel already in `polytopia.py`.
3. **Phase 3 — Break pta dependency** from `ThermoPrior` — `pta` requires Gurobi + Docker, not pip-installable. Replace with pure-Python equivalents.
4. **Phase 4 — Code quality** — fix known bugs, type hints, pin deps, implement MLE.
5. **Phase 5 — SBI/CNF pipeline** — high-level API wrapping the existing CNF infrastructure.
6. **Phase 6 — Tests** — expand coverage to untested modules.
7. **Phase 7 — Docs** — docstrings, Sphinx, tutorial notebooks.

Dependencies have no pinned versions — address during Phase 3 cleanup.

---

## Phase 1 — Delete LinAlg entirely / torch-only migration

**Goal:** Delete `linalg.py` entirely. Replace every `self._la.X(...)` call with direct `torch.X(...)`. Move the handful of genuinely project-specific helpers to `core/util.py` and a new `core/distributions.py`. Handle reproducibility via `torch.Generator` passed explicitly.

### Background

`LinAlg` (`src/sbmfi/core/linalg.py`, ~1200 lines) abstracts NumPy/SciPy vs PyTorch behind a `backend` string. Every major class stores `self._la: LinAlg`. With torch ≥ 2.0 a hard dependency and NumPy support dead weight, the class has no reason to exist. Everything it does is either a direct `torch` call or one of a few small project-specific helpers.

### What moves where

| Current `LinAlg` feature | Replacement |
|---|---|
| ~50 delegated torch functions (`exp`, `einsum`, `cat`, …) | Direct `torch.X(...)` calls inline |
| `get_tensor(shape, indices, values, …)` | Standalone function in `core/util.py` — builds dense tensor from sparse indices, merges duplicates |
| `_merge_duplicate_indices()` | Private helper alongside `get_tensor()` in `core/util.py` |
| `convolve(a, v)` | Standalone `convolve(a, v)` in `core/util.py` via `torch.nn.functional.conv1d`. Used heavily in EMU simulation. |
| `norm_* / trunc_norm_*` probability functions | Standalone functions in new `core/distributions.py` built on `torch.distributions.Normal` |
| `tonp(x)` | `x.detach().cpu().numpy()` inline at API boundaries |
| `vecopy(A)` | `A.clone()` |
| `_batch_size` attribute | Parameter of `Simulator.__init__` directly |
| `seed` / RNG state | `torch.Generator` passed explicitly via `@with_generator` decorator |
| `device` | Passed to constructors that create tensors; default `'cpu'` |

### RNG pattern — `@with_generator` decorator in `core/util.py`

```python
def with_generator(fn):
    @functools.wraps(fn)
    def wrapper(*args, generator=None, **kwargs):
        if generator is None:
            generator = torch.Generator()
        return fn(*args, generator=generator, **kwargs)
    return wrapper
```

Applied to any function calling `torch.randn`, `torch.rand`, etc. Classes that sample repeatedly (priors, polytope samplers) store a `torch.Generator` on `self`. Callers needing reproducibility pass `torch.Generator().manual_seed(seed)`; others omit it.

### Truncated normal implementation (`core/distributions.py`)

```python
_U = Normal(torch.tensor(0.0, dtype=torch.double),
            torch.tensor(1.0, dtype=torch.double))

def trunc_norm_log_pdf(x, mu, sigma, lo, hi):
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    return Normal(mu, sigma).log_prob(x) - torch.log(_U.cdf(b) - _U.cdf(a))

def trunc_norm_cdf(x, mu, sigma, lo, hi):
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    Z = _U.cdf(b) - _U.cdf(a)
    return (_U.cdf((x - mu) / sigma) - _U.cdf(a)) / Z

def trunc_norm_inv_cdf(p, mu, sigma, lo, hi):
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    return mu + sigma * _U.icdf(_U.cdf(a) + p * (_U.cdf(b) - _U.cdf(a)))

def trunc_norm_pdf(x, mu, sigma, lo, hi):
    return trunc_norm_log_pdf(x, mu, sigma, lo, hi).exp()
```

### Steps

- [ ] **1.1** Create `src/sbmfi/core/distributions.py` with the four `trunc_norm_*` and four `norm_*` functions.
- [ ] **1.2** Add `with_generator` decorator, `get_tensor()`, `_merge_duplicate_indices()`, and `convolve()` to `src/sbmfi/core/util.py`.
- [ ] **1.3** Replace all `self._la.tonp(x)` calls with `x.detach().cpu().numpy()`. Grep target: `\.tonp\(`.
- [ ] **1.4** Replace all `self._la.X(...)` calls in each module with direct `torch.X(...)` or the new util/distributions functions, module by module:
  - [ ] `core/coordinater.py`
  - [ ] `core/model.py`
  - [ ] `core/observation.py`
  - [ ] `core/polytopia.py`
  - [ ] `core/simulator.py` — move `_batch_size` here from LinAlg
  - [ ] `core/simulfuncs.py`
  - [ ] `inference/sampling.py`
  - [ ] `priors/uniform.py` — use `distributions.py`; store `torch.Generator` on `self`
  - [ ] `priors/projection_prior.py`, `ratio_prior.py`, `mog.py`
- [ ] **1.5** Replace `LinAlg(...)` construction sites in `models/small_models.py`, `models/build_models.py`, `tests/conftest.py` with `torch.Generator().manual_seed(seed)` where seed matters; remove otherwise.
- [ ] **1.6** Delete `self._la` attribute and `linalg` constructor parameter from every class once its module is migrated.
- [ ] **1.7** Delete `src/sbmfi/core/linalg.py` entirely.
- [ ] **1.8** Delete `tests/test_core/test_linalg.py`; add tests for `get_tensor()`, `convolve()`, and `trunc_norm_*` to `tests/test_core/test_util.py` and new `tests/test_core/test_distributions.py`.
- [ ] **1.9** Run full test suite and fix regressions.

**Files deleted:** `core/linalg.py`, `tests/test_core/test_linalg.py`  
**Files created:** `core/distributions.py`  
**Files modified:** `core/util.py`, `core/model.py`, `core/observation.py`, `core/coordinater.py`, `core/polytopia.py`, `core/simulator.py`, `core/simulfuncs.py`, `inference/sampling.py`, `priors/uniform.py`, `priors/projection_prior.py`, `priors/ratio_prior.py`, `priors/mog.py`, `models/small_models.py`, `models/build_models.py`, `tests/conftest.py`

---

## Phase 2 — MCMC / ABC refactor

**Goal:** Replace the broken MCMC implementation in `sampling.py` with a clean one that delegates to the `MarkovTransition` kernel already implemented in `polytopia.py`. Make both Peskun and Barker multi-proposal hit-and-run work correctly for exact posteriors and ABC distance kernels.

### Background

The paper (Appendix B, Algorithm 1) describes a **multi-proposal hit-and-run** sampler:
- At each step, sample a random direction `s` from the unit sphere.
- Compute chord extremes `α_min`, `α_max` (the intersection of the ray with the polytope boundary).
- Draw M proposals: `α_i ~ q(α; α_min, α_max)` (uniform or truncated-normal).
- Evaluate target density π at the current point and all M candidates.
- Accept one candidate via **Peskun** (eq. 60, lower asymptotic variance) or **Barker** (eq. 61, softmax) weights.

**What exists and works:**
- `MarkovTransition` in `core/polytopia.py` (lines 883–1005): implements Algorithm 1 correctly for both Barker and Peskun. It takes a `target_density` with a `.log_prob()` method and handles the full multi-candidate acceptance.
- `sample_polytope()` in `core/polytopia.py` accepts an optional `markov_transition=MarkovTransition(...)` argument and already calls it correctly for non-uniform densities.

**What is broken:**
- `MCMC.accept_reject()` in `sampling.py` (lines 685–795): a *separate, redundant* reimplementation of Peskun that is incomplete and raises `NotImplementedError` at line 795.
- `_BaseBayes.perturb_particles()` (lines 375–442): generates chord candidates but does not apply acceptance, duplicating the chord-extremes logic already in `MarkovTransition`.
- `DistanceKernel` class (lines 601–642): broken due to undefined attribute references (`self._x_meas`, `self._fcm`, etc.) — copy-paste errors from `_BaseBayes`.
- `SMC` particle perturbation step uses `perturb_particles()` directly, bypassing `MarkovTransition`.

**The fix:** `MCMC.run()` should construct a `MarkovTransition(target_density=self.log_prob, ...)` and call `sample_polytope(..., markov_transition=transition)`. This makes the sampler correct-by-construction and eliminates ~300 lines of broken code.

### Steps

- [ ] **2.1** Fix `DistanceKernel` in `sampling.py`: replace broken attribute references with the correct ones from `_BaseBayes` (`self._m.compute_distance()`, `self._m._x_meas`, etc.). This is needed for ABC mode.
- [ ] **2.2** Add a `log_prob` wrapper that satisfies the `target_density` interface expected by `MarkovTransition`:
  - For exact MCMC: `target_density.log_prob(theta)` calls `_BaseBayes.log_prob(theta)` (posterior).
  - For ABC: `target_density.log_prob(theta)` returns the negative distance (so smaller distance = higher "probability") via `DistanceKernel`.
- [ ] **2.3** Rewrite `MCMC.run()` to:
  - Construct `MarkovTransition(model=self._sampler, target_density=..., n_cdf=M, proposal_id=..., chord_std=..., transition_id='peskun'|'barker')`.
  - Call `sample_polytope(model=self._sampler, n=N, n_chains=L, markov_transition=transition, ...)`.
  - Collect output and wrap in `arviz.InferenceData` (keep the existing output formatting).
- [ ] **2.4** Delete `MCMC.accept_reject()` (lines 685–795, the broken Peskun reimplementation).
- [ ] **2.5** Delete or simplify `_BaseBayes.perturb_particles()` — keep only the exchange-flux perturbation logic (lines 422–433) which is not handled by `MarkovTransition`.
- [ ] **2.6** Rewrite `SMC._sample_next_population()` to use `MarkovTransition` for the chord-based particle perturbation step, replacing the `perturb_particles()` call. Exchange fluxes continue to be perturbed independently.
- [ ] **2.7** Verify `MarkovTransition` Peskun path is complete for `n_cdf > 2` (the exploration suggests it is, but double-check against eq. 60 from the paper).
- [ ] **2.8** Run the existing toy-model smoke test (`spiro` model) for both exact MCMC and ABC-SMC. Check R-hat and ESS via arviz match the paper's Table 2 ballpark figures (R-hat ≈ 1.0, ESS > 10%).
- [ ] **2.9** Write tests: `tests/test_inference/test_sampling.py` — one MCMC run on `spiro` (Peskun, M=3) and one SMC/ABC run.

**Files modified:** `inference/sampling.py`, `core/polytopia.py`  
**Lines deleted:** `MCMC.accept_reject()` (~110 lines), bulk of `_BaseBayes.perturb_particles()` (~70 lines), `DistanceKernel` rewrite  
**Key reuse:** `MarkovTransition.__call__()`, `sample_polytope()`, existing arviz output formatting

---

## Phase 3 — Replace pta's dG sampler with pure Python/torch

**Goal:** Keep pta as a dependency for model construction (structural assessment, TFSModel building, ConcentrationsPrior) — that is fine and stays. Only replace the **dG sampling** step (`sample_drg`, `get_initial_points`, `PmoProblemPool`, `_find_point`) which relies on C++ dependencies that don't work on Windows.

### Background

`pta` is used in two distinct ways:

1. **Model construction** (Docker / offline): `pta.StructuralAssessment`, `pta.FluxSpace`, `pta.ThermodynamicSpace`, `pta.PmoProblem`, `pta.TFSModel`, `pta.ConcentrationsPrior` — all stay as-is. Users build and pickle the `TFSModel` via the Docker workflow. The pickle loads fine on any platform as long as pta is installed.

2. **dG sampling** (`priors/thermo_prior.py` online): `sample_drg()`, `PmoProblemPool`, `_find_point` — these call into pta's C++ sampler, which breaks on Windows. **This is the only part we replace.**

What we replace vs. what stays:

| Symbol | Status | Action |
|---|---|---|
| `TFSModel`, `ConcentrationsPrior`, `FluxSpace`, etc. | **Keep** — no C++ at runtime | No change |
| `FreeEnergiesSamplingResult` | **Replace** — return type of `sample_drg` | Own `DRGSamplingResult` dataclass |
| `sample_drg()` | **Replace** — C++ MCMC sampler | Pure Python hit-and-run on the MVN |
| `_find_point`, `PmoProblemPool` | **Replace** — C++ Gurobi LP solver | `scipy.optimize.linprog` |
| `R` constant (pint units) | **Replace** — pint causes pickling issues | `R_GAS = 8.314472e-3  # kJ mol⁻¹ K⁻¹` |

Key insight: `ThermoPrior.extract_drg_mvn()` already extracts the full MVN (mean + covariance) from `TFSModel`. `sample_drg()` then just samples from this MVN truncated to an orthant (sign constraints from flux directions). This is standard truncated MVN sampling — replaceable with a pure Python hit-and-run.

### Session Status — 2026-04-14

- [x] Added and extended `src/sbmfi/priors/thermo_data.py` as a portable container for thermodynamic sampling inputs.
- [x] `ThermoSamplingData.from_tfs_model(...)` now extracts the PTA basis-to-dG transform, confidence radius, irreversible thermodynamic inequalities, constrained reaction indices, and full flux stoichiometry/bounds.
- [x] Updated `src/sbmfi/priors/thermo_prior.py` to use the pure-Python thermo sampler path instead of PTA runtime sampling, and fixed cache-filling / pandas integration issues so the prior runs again.
- [x] Added thermo tests in `src/sbmfi/tests/test_priors/test_thermo.py`; these passed for the current simplified sampler.
- [x] Fixed solver consistency so PolyRound now follows the configured Cobra LP solver instead of assuming Gurobi. Relevant changes were made in `src/sbmfi/config.py`, `src/sbmfi/core/polytopia.py`, and `src/sbmfi/models/build_models.py`.
- [x] Verified from the local PTA clone in `C:\\python_projects\\pta` that the real PTA sampler works in basis space and performs ray segmentation plus steady-state feasibility checks.
- [!] `src/sbmfi/priors/thermo_sampler.py` is still the old simplified dG-space truncated-MVN sampler. It has not yet been rewritten to match PTA Algorithm 1.
- [ ] Remaining Phase 3 work: rewrite `thermo_sampler.py` around the extracted basis-space variables, then re-run thermo tests and the `ThermoPrior` smoke test.

### Steps

- [ ] **3.1** Create `src/sbmfi/priors/thermo_sampler.py`:
  - `DRGSamplingResult` dataclass: `.samples: pd.DataFrame`, `.basis_samples: pd.DataFrame` — same shape/columns as the pta equivalent used in `thermo_prior.py`.
  - `find_initial_points(tfs_model: TFSModel, n: int) -> np.ndarray`: replaces the `TFSModel.get_initial_points` monkey-patch. Uses `scipy.optimize.linprog` to find feasible starting points in dG space. Removes dependency on `PmoProblemPool` and `_find_point`.
  - `sample_drg(tfs_model: TFSModel, initial_points, num_samples, num_chains, ...) -> DRGSamplingResult`: hit-and-run MCMC for the orthant-constrained MVN. The MVN parameters are extracted inline (same logic as `extract_drg_mvn()`). PSRF convergence check via `arviz`.

- [ ] **3.2** Update `priors/thermo_prior.py`:
  - Replace `from pta.sampling.tfs import sample_drg, FreeEnergiesSamplingResult, _find_point, PmoProblemPool` with imports from `thermo_sampler.py`.
  - Replace `from pta.constants import R` with `R_GAS = 8.314472e-3`.
  - Remove the `TFSModel.get_initial_points` monkey-patch at module level (lines 20–68) — replaced by `find_initial_points()` in `thermo_sampler.py`.
  - Update `_sample_drg_suppress_output()` to call the new `sample_drg()`.
  - Keep all `TFSModel` type annotations and all `tfs_model.T.*` / `tfs_model.F.*` attribute access unchanged.

- [ ] **3.3** Complete `ThermoPrior.log_prob()` for `'thermo'` coordinates:
  - Evaluate log probability under the truncated MVN.
  - Orthant normalization constant via Monte Carlo approximation.
  - Keep `NotImplementedError` for `'labelling'` coordinates.

- [ ] **3.4** Write tests: `sample_drg` on a small synthetic MVN (2–3 reactions), check output shape, column names, and PSRF < 1.1.

**Files created:** `priors/thermo_sampler.py`  
**Files modified:** `priors/thermo_prior.py`  
**Unchanged:** `TFSModel` usage, `ConcentrationsPrior`, `models/mixed_substrate_experiment.py`, all Docker workflow files

---

## Phase 4 — Core Code Quality

**Goal:** Fix known bugs, add type hints, pin dependencies.

- [ ] **3.1** Fix charge handling in `core/metabolite.py` (TODO: "CHARGES ARE NOT REGISTERED CORRECTLY!").
- [ ] **3.2** Fix natural abundance correction in `core/observation.py` (TODO: "this currently sucks", line ~40).
- [ ] **3.3** Add type hints to all public classes and methods in `core/`.
- [ ] **3.4** Pin major dependency versions in `pyproject.toml`:
  - `torch>=2.0,<3`, `numpy>=1.24`, `scipy>=1.10`, `cobra>=0.29`, `PolyRound>=0.3`, `pandas>=2.0`
  - Add `python_requires=">=3.10"`.
- [ ] **3.5** Implement `inference/mle.py`: SQP-based MLE using `scipy.optimize.minimize` + SLSQP with the Jacobian from `EMU_Model.compute_jacobian()`. Useful for paper comparison with Bayesian/SBI results.
- [ ] **3.6** Audit and resolve TODO/FIXME comments throughout.
- [ ] **3.7** Run `ruff check --fix` and `black` across `src/sbmfi/`.
- [ ] **3.8** Rename `obervervator_worker` → `observator_worker` in `simulfuncs.py` (typo).

**Files primarily affected:** `core/metabolite.py`, `core/observation.py`, `pyproject.toml`, `inference/mle.py`, `core/simulfuncs.py`

---

## Phase 5 — SBI / CNF Pipeline

**Goal:** Make MCMC/SMC/ABC production-quality; build a high-level SBI pipeline using the existing CNF infrastructure.

### 4A — MCMC / SMC / ABC Polish

MCMC (`sampling.py:MCMC`) and SMC/ABC (`sampling.py:SMC`) are structurally complete but need:

- [ ] **4A.1** Verify Peskun-optimality conditions in `MCMC._step()` acceptance logic.
- [ ] **4A.2** Add `MCMC.diagnostics()` returning arviz `InferenceData` with ESS, R-hat, trace plots.
- [ ] **4A.3** Smoke-test `SMC` epsilon schedule on the `spiro` toy model.
- [ ] **4A.4** Add `DataSetSim.to_xarray()` for arviz interop.
- [ ] **4A.5** Write tests: `test_inference/test_sampling.py` — one MCMC run and one SMC/ABC run on `spiro`.

### 4B — Simulation-Based Inference with CNF

Infrastructure in `continuous_flows.py` and `discrete_flows.py` exists. Missing: a high-level API connecting simulations → training data → trained flow → posterior.

- [ ] **4B.1** Create `inference/sbi_pipeline.py`:
  - `SimulationDataset`: wraps `DataSetSim` into a `torch.utils.data.Dataset` of `(theta, x_obs)` pairs.
  - `train_flow(simulator, prior, n_simulations, flow_type='continuous'|'discrete')`: orchestrates simulation, training, returns a trained flow. **Default: `'continuous'`** via `continuous_flows.py` + `flow_matching`.
  - `posterior_samples(flow, x_obs, n_samples)`: posterior samples from a trained flow.
- [ ] **4B.2** Primary backend: `riem_sample_and_div()` from `continuous_flows.py`. Discrete path: `flow_constructor()` / `flow_trainer()` from `discrete_flows.py`.
- [ ] **4B.3** Demo notebook: `docs/notebooks/sbi_cnf_demo.ipynb` — end-to-end SBI on `spiro`.
- [ ] **4B.4** Test: train flow on 500 `spiro` simulations; check posterior shape and finite log-prob.

**Files primarily affected:** `inference/sampling.py`, new `inference/sbi_pipeline.py`, `inference/discrete_flows.py`, `inference/continuous_flows.py`

---

## Phase 6 — Test Coverage

- [ ] **5.1** `test_core/test_observation.py` — `MDV_ObservationModel` construction, `log_lik()` shape, transforms.
- [ ] **5.2** `test_core/test_simulator.py` — `Simulator` MDV shapes for `spiro`.
- [ ] **5.3** `test_core/test_simulfuncs.py` — worker functions with mock model.
- [ ] **5.4** `test_core/test_util.py` — regex patterns and `make_multidex`.
- [ ] **5.5** `test_inference/test_sampling.py` — MCMC + SMC smoke tests (see 4A.5).
- [ ] **5.6** `test_priors/test_uniform.py` — `UniFluxPrior` sample shapes and cache filling.
- [ ] **5.7** `test_priors/test_thermo.py` — `ThermoSamplingData` round-trip and sampler output.
- [ ] **5.8** Add coverage reporting to CI: fail below 60% line coverage.

---

## Phase 7 — Docstrings and Documentation

- [ ] **6.1** NumPy-style docstrings for all public classes/methods in `core/`.
- [ ] **6.2** Docstrings for `inference/` public APIs.
- [ ] **6.3** Docstrings for `priors/` public APIs.
- [ ] **6.4** Fix broken example cells in `docs/notebooks/introduction.ipynb`.
- [ ] **6.5** Add `docs/notebooks/priors_and_sampling.ipynb` — `UniFluxPrior`, `ThermoPrior`, MCMC/SMC.
- [ ] **6.6** Sphinx build passes cleanly with no errors.
- [ ] **6.7** Update `README.md`: description, installation, quick-start snippet, paper link.

---

## File Map

| Path | Role | Phases |
|---|---|---|
| `src/sbmfi/core/linalg.py` | LinAlg abstraction — delete most | 1 |
| `src/sbmfi/core/model.py` | LabellingModel, EMU_Model | 1, 3 |
| `src/sbmfi/core/observation.py` | MDV_ObservationModel | 1, 3, 5 |
| `src/sbmfi/core/coordinater.py` | FluxCoordinateMapper | 1 |
| `src/sbmfi/core/polytopia.py` | Polytope sampling | 1 |
| `src/sbmfi/core/simulator.py` | Simulator | 1, 5 |
| `src/sbmfi/core/simulfuncs.py` | Batch simulation workers | 1, 3, 5 |
| `src/sbmfi/core/metabolite.py` | Metabolite classes | 3 |
| `src/sbmfi/inference/sampling.py` | MCMC, SMC, ABC | 1, 4A |
| `src/sbmfi/inference/discrete_flows.py` | Neural spline flows | 4B |
| `src/sbmfi/inference/continuous_flows.py` | ODE/CNF flows | 4B |
| `src/sbmfi/inference/mle.py` | MLE stub — implement | 3 |
| `src/sbmfi/inference/sbi_pipeline.py` | **New** — SBI high-level API | 4B |
| `src/sbmfi/priors/thermo_prior.py` | ThermoPrior | 2 |
| `src/sbmfi/priors/thermo_data.py` | **New** — ThermoSamplingData | 2 |
| `src/sbmfi/priors/thermo_sampler.py` | **New** — pure-Python dG sampler | 2 |
| `src/sbmfi/priors/uniform.py` | Flux priors | 1 |
| `src/sbmfi/models/mixed_substrate_experiment.py` | Model builders | 2 |
| `src/sbmfi/models/small_models.py` | Toy model builders | 1 |
| `pyproject.toml` | Dependencies | 3 |

---

## Decisions

1. **MLE module** → Implement using `scipy.optimize.minimize` + SLSQP + `EMU_Model.compute_jacobian()`.
2. **SBI flow type default** → Continuous (CNF/ODE). Discrete available as `flow_type='discrete'`.
3. **pta dependency** → Keep pta for model construction (TFSModel, ConcentrationsPrior, etc.) — no change there. Only replace `sample_drg()`, `PmoProblemPool`, and `_find_point` with pure Python/torch, because the C++ backend breaks on Windows.
4. **ConcentrationsPrior** → Keep as-is via pta.

---

## Architecture Review — 2026-04-14

### Overall assessment

The codebase is technically rich and clearly written by someone who understands the domain, but it still has the shape of a research codebase that grew by accretion. The main issue is not lack of capability; it is that too many modules own too many responsibilities at once. That makes the project feel dated even when the underlying ideas are good.

The strongest improvement opportunity is to make the architecture more explicit:
- domain objects should describe the metabolic system and transformations,
- numerical backends should do linear algebra / optimization / sampling,
- inference code should orchestrate priors, simulators, and likelihoods,
- builders should assemble configured objects, not contain data cleaning, parsing, analysis, and experiments in one file.

Shorter and more readable here mostly means reducing role-mixing and duplicate pathways, not micro-optimizing syntax.

### Main findings

1. **A few files are carrying too much system complexity.**
   `src/sbmfi/core/model.py`, `src/sbmfi/inference/sampling.py`, `src/sbmfi/core/polytopia.py`, `src/sbmfi/core/observation.py`, `src/sbmfi/core/coordinater.py`, and `src/sbmfi/models/build_models.py` are all large enough to hide multiple sub-architectures inside one module.

2. **The core abstraction boundaries are blurry.**
   `LabellingModel` is not just a model. It also owns solver configuration, build lifecycle, flux bookkeeping, substrate labeling state, and EMU-related behavior through subclasses.
   `_BaseBayes` is not just inference scaffolding. It also acts as simulator wrapper, data holder, posterior evaluator, and measurement registry.
   `FluxCoordinateMapper` is not just a mapper. It contains many transforms, coordinate policy decisions, and thermo/labelling glue.

3. **There is too much mixing of tensor code and dataframe code inside hot paths.**
   Many APIs accept `torch.Tensor`, `np.ndarray`, or `pd.DataFrame` interchangeably and convert back and forth in the middle of computational logic. That is convenient locally but expensive cognitively and often physically.
   This especially affects `coordinater.py`, `simulator.py`, `sampling.py`, and `thermo_prior.py`.

4. **Runtime monkey-patching is a real maintainability smell.**
   `thermo_prior.py` mutates `FluxCoordinateMapper` at import time. That makes behavior non-local, harder to test, and harder to reason about.
   If a method belongs on `FluxCoordinateMapper`, it should live there or on a dedicated adapter object.

5. **Library code still contains research-era debug behavior.**
   There are many `print(...)` calls, partially finished TODO blocks, commented exploratory code, and module-level experimentation fragments.
   That makes it harder to trust whether a file is “library code” or “scratchpad that still happens to run”.

6. **Builders and experiments are too entangled.**
   `build_models.py` looks like a mix of:
   - model factory layer,
   - dataset parsing,
   - Anton/Tomek specific ETL,
   - observation-model assembly,
   - one-off correction logic,
   - and experiment recipes.
   This file should be factored into separate modules by responsibility.

7. **The geometry / optimization layer is powerful but too monolithic.**
   `polytopia.py` contains polytope definitions, cdd utilities, projection, vertex conversion, null-space helpers, rounding, sampling models, Markov transition logic, and volume estimation.
   These are related, but they are not one responsibility.

8. **Naming and layering still reflect implementation history instead of stable concepts.**
   Examples:
   - `_BaseBayes`, `BaseRoundedPrior`, `_BaseXchFluxPrior`, `_CannonicalPolytopeSupport`
   - `coordinater.py`
   - model classes that accumulate mixin behavior and side effects
   The code works, but the module/class names do not yet communicate a clean product architecture.

### What to factor out

- **From `core/model.py`**
  - Extract model construction / mutation helpers into `core/model_builder.py` or `core/model_construction.py`.
  - Extract substrate-labelling state management into a dedicated component.
  - Extract EMU compilation/build artifacts into a separate `EMUCompiler` or `EMUBuilder`.
  - Keep `LabellingModel` itself focused on domain state and validated operations.

- **From `core/coordinater.py`**
  - Split pure coordinate transforms from mapper orchestration.
  - A good split would be:
    - `core/coordinate_transforms.py` for stateless `map_*` functions,
    - `core/flux_mapper.py` for the stateful `FluxCoordinateMapper`,
    - `core/thermo_mapping.py` for Gibbs/thermo-specific helpers.
  - Move free functions that are only used by thermo code out of the generic coordinate module.

- **From `core/polytopia.py`**
  - Split into:
    - `core/polytope_ops.py` for H/V-representation and projection utilities,
    - `core/polytope_rounding.py` for PolyRound integration,
    - `core/polytope_sampling.py` for `PolytopeSamplingModel`, `MarkovTransition`, `sample_polytope`,
    - `core/polytope_volume.py` if volume estimation stays.
  - This one change would improve readability a lot because `polytopia.py` currently hides several distinct subsystems.

- **From `inference/sampling.py`**
  - Separate exact posterior inference from ABC/SMC code.
  - A likely split:
    - `inference/base.py` for common posterior/simulator scaffolding,
    - `inference/mcmc.py`,
    - `inference/smc.py`,
    - `inference/potentials.py` for `_PosteriorDensity` and distance kernels.
  - `_BaseBayes` should probably become a public, better-named orchestration object, or be reduced sharply.

- **From `models/build_models.py`**
  - Split by dataset / source:
    - `models/anton.py`
    - `models/tomek.py`
    - `models/ecoli_core.py`
  - Split ETL/parsing from factory functions:
    - `models/parsers/anton.py`
    - `models/factories.py`
  - Keep `build_models.py` only as a thin compatibility facade if needed.

- **From `core/observation.py`**
  - Separate observation definitions from statistical noise models and from data-frame munging.
  - The current file combines:
    - observation indexing / schema logic,
    - transforms,
    - Gaussian block likelihood machinery,
    - concrete LCMS/classical observation models.

### What to factor in

- **Introduce small typed config/state objects**
  - `SBMFIConfig` is a good start.
  - Add explicit dataclasses for things like:
    - sampling settings,
    - observation schema,
    - thermo sampling payload,
    - simulation batch settings,
    - model build options.
  - This will reduce huge function signatures and undocumented implicit state.

- **Introduce an internal “data boundary” convention**
  - Decide which layers operate on `torch.Tensor` only.
  - Convert to/from pandas only at API boundaries, diagnostics, and persistence layers.
  - This single rule would simplify many functions.

- **Introduce explicit adapter classes instead of monkey-patches**
  - Example: thermo-specific behavior on top of `FluxCoordinateMapper` should be a `ThermoFluxAdapter` or similar, not import-time mutation.

- **Introduce one logging policy**
  - Replace raw `print(...)` in library code with `logging`.
  - Keep notebook/demo output separate from core library behavior.

### Concrete refactoring priorities

1. **Kill runtime monkey-patching and hidden behavior injection.**
   Highest-value cleanup because it improves locality immediately.

2. **Split `build_models.py` by dataset/source and move parsing out of factories.**
   This will remove a lot of perceived age and clutter from the repo.

3. **Split `polytopia.py` into geometry, rounding, and sampling modules.**
   This should make the sampling architecture much easier to understand and maintain.

4. **Enforce tensor-first internals, dataframe-at-the-edge.**
   This will shorten many functions and remove a lot of branchy conversion code.

5. **Break `inference/sampling.py` into exact MCMC, SMC/ABC, and shared posterior utilities.**
   The current module is too broad to stay healthy.

6. **Rename and stabilize core abstractions.**
   Examples:
   - `coordinater.py` → `flux_mapper.py` or `coordinates.py`
   - `_BaseBayes` → `PosteriorProblem` or `InferenceProblem`
   - `_CannonicalPolytopeSupport` → `CanonicalPolytopeSupport`

7. **Remove dead comments, exploratory blocks, and prints from production modules.**
   This is not cosmetic. It makes it easier to see the real design.

### Engineering style guidance for future changes

- Prefer one canonical internal representation per layer.
- Prefer composition over inheritance when adding thermo, ratio, or observation-specific behavior.
- Keep module-level files below the point where they need their own table of contents.
- If a function accepts pandas, numpy, and torch, that is usually a sign the boundary is in the wrong place.
- If a class both stores domain state and orchestrates optimization/sampling, split it.
- If behavior is added by import side effects, move it into an explicit object or method.

### Suggested new medium-term phase

- [ ] **Architecture cleanup phase**
  - [ ] Split `build_models.py` into source-specific parser/factory modules.
  - [ ] Split `polytopia.py` into geometry / rounding / sampling modules.
  - [ ] Replace thermo-related `FluxCoordinateMapper` monkey-patching with explicit methods or an adapter.
  - [ ] Define tensor-only internal interfaces for simulation, priors, and inference.
  - [ ] Move statistical likelihood code out of large observation/model classes into smaller composable components.
  - [ ] Standardize logging and remove `print(...)` from library code.
  - [ ] Rename historically-grown modules/classes to match stable concepts.

---

## Execution Log

| Date | Phase | Notes |
|---|---|---|
| 2026-04-13 | — | Plan written after full codebase exploration |
| 2026-04-14 | 3 | Added portable thermodynamic extraction in `src/sbmfi/priors/thermo_data.py` and switched `ThermoPrior` to the pure-Python thermo sampler path. |
| 2026-04-14 | 3 | Added thermo tests and verified the current simplified sampler/prior path works on a toy model under an open-source LP backend. |
| 2026-04-14 | 3 | Fixed Cobra/PolyRound solver alignment so PolyRound uses the same configured LP backend instead of assuming Gurobi. |
| 2026-04-14 | 3 | Reverse engineered the actual PTA steady-state free-energy sampler from `C:\\python_projects\\pta` and confirmed `src/sbmfi/priors/thermo_sampler.py` still needs a full basis-space rewrite. |
