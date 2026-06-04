# Plan: Pure-Python Thermodynamic Flux Sampling

**Goal**: Implement Thermodynamics and Flux Sampling (TFS, Gollub et al. 2020) entirely
in Python — no C++ binaries, no Gurobi, no Docker — while reusing pta's math
where it matters and keeping the code clean and well-tested.

## Background

The paper defines the *steady-state thermodynamic space* T as:

```
ΔrG' = ΔrG'° + RT · Sᵀ · ln c            (Eq. 2)
ln c  ~ N(μ_c, Σ_c)                        (Eq. 3)
ΔrG'° ~ N(μ_o, Σ_o)                        (Eq. 4)
```

The joint vector `t = [ln c, ΔrG'°, ΔrG']` is MVN with covariance Σ_t (Eq. 6).
Because Σ_t is often rank-deficient (correlated group-contribution estimates), a
reduced basis is used:

```
t = μ_t + Q · m,   m ~ N(0, I_q)           (Eq. 8)
```

where Q = Σ_t^{1/2} and q = rank(Σ_t).  For sampling we only need the marginal
of ΔrG', so we store just the sub-block of the transform:

```
ΔrG' = T · m + s,  m ~ N(0, I_q)
T ∈ R^{γ×q},  s ∈ R^γ
```

The second law forces `v_i · ΔrG'_i < 0` for each reaction in Γ (Eq. 9).
Feasibility of flux then depends on whether the current *orthant*
(sign pattern of ΔrG') admits a steady-state flux solution (Eq. 1).

TFS (Algorithm 1) is a hit-and-run sampler that:
1. Works in the isotropic m-space (the ellipsoid becomes a sphere `||m||² ≤ r²`)
2. At each step enumerates all line-segment intersections with the feasible region
3. Checks LP feasibility for each orthant encountered on the line
4. Weights valid segments by their MVN probability mass (CDF differences)
5. Tracks orthant visit counts → orthant probabilities

**Current state of `thermo_sampler.py`**: implements hit-and-run in dG space
directly, without the basis parametrisation, without the ellipsoidal bound,
and without orthant-feasibility checking.  It produces dG samples but not
orthant statistics, and mixing is poor because the space is non-isotropic.

---

## Step 1 — `pta_port/`: pure-Python pta essentials

**What & why**: Port the three pta pieces that cannot be installed on Windows:
`covariance_square_root`, `apply_transform`, and `ThermodynamicSpaceBasis`
(the function that builds T and s from raw model parameters).  No cobra, no
enkie, no component_contribution, no C++.

### Files to create

```
src/sbmfi/priors/pta_port/
    __init__.py
    linalg.py          — qr_rank_deficient() via scipy (replaces component_contribution)
    utils.py           — covariance_square_root(), apply_transform()
    thermo_space.py    — ThermodynamicSpaceParams (numpy dataclass)
                         build_drg_basis(params) → (T, s, q)
    flux_space.py      — FluxSpaceParams (numpy dataclass: S, lb, ub, ids)
```

### `pta_port/linalg.py`

`LINALG.qr_rank_deficient(A)` in component_contribution performs a thin QR of
matrix A and drops near-zero columns (Gram-Schmidt with threshold).  Replace with:

```python
def qr_rank_deficient(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Thin QR of A, dropping columns whose diagonal R entry is < tol * max(diag R).
    
    Used the same way pta uses component_contribution.linalg.LINALG.qr_rank_deficient:
        result = qr_rank_deficient(X.T).T
    where X is the observables covariance square root.
    """
    Q, R, P = scipy.linalg.qr(A, pivoting=True, mode='economic')
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * diag[0])) if diag[0] != 0 else 0
    # Reorder columns back (undo pivoting) and return thin result
    R_thin = R[:rank, :]
    R_thin[:, P] = R_thin.copy()          # undo column permutation
    return Q[:, :rank] @ R_thin
```

Unit test: verify `qr_rank_deficient(A.T).T @ qr_rank_deficient(A.T)` ≈ `A.T @ A`
for a rank-deficient `A` built from known group-contribution data.

### `pta_port/utils.py`

Direct port of `pta.utils.covariance_square_root` and `apply_transform`
(these are pure numpy already — ~20 lines total).

### `pta_port/thermo_space.py`

```python
@dataclass
class ThermodynamicSpaceParams:
    """All raw parameters needed to build the dG basis.  No enkie objects."""
    reaction_ids:       list[str]
    metabolite_ids:     list[str]
    T_kelvin:           float
    dfg0_prime_mean:    np.ndarray          # (n_met,)
    dfg0_prime_cov_sqrt: np.ndarray         # (n_met, q_dfg)
    log_conc_mean:      np.ndarray          # (n_met,)
    log_conc_cov:       np.ndarray          # (n_met, n_met)
    S_constraints:      np.ndarray          # (n_met, n_rxn)


def build_drg_basis(
    params: ThermodynamicSpaceParams,
    min_eigenvalue: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the affine map ΔrG' = T·m + s, m ~ N(0, I_q).

    Returns
    -------
    T : (n_rxn, q) — columns are the principal directions
    s : (n_rxn,)   — mean vector of ΔrG'
    q : int        — effective rank (dimensionality of m)
    """
```

This is a direct port of `ThermodynamicSpaceBasis._compute_basis()` with
`explicit_drg=True, explicit_drg0=False, explicit_log_conc=False`.
The only nontrivial step is replacing `LINALG.qr_rank_deficient` with our
`pta_port.linalg.qr_rank_deficient`.

### `pta_port/flux_space.py`

```python
@dataclass
class FluxSpaceParams:
    S:            np.ndarray    # (n_met, n_rxn)
    lb:           np.ndarray    # (n_rxn,)
    ub:           np.ndarray    # (n_rxn,)
    reaction_ids: list[str]
    metabolite_ids: list[str]
```

No logic — just a typed container.  Add `from_cobrapy_model(model)` as
an optional classmethod (keeps cobra optional).

### Tests: `test_priors/test_pta_port.py`

- **linalg**: `qr_rank_deficient` on synthetic rank-deficient matrices; compare
  Gram to input Gram up to numerical noise.
- **utils**: `covariance_square_root` roundtrip `C_sqrt @ C_sqrt.T ≈ C`.
- **build_drg_basis** (toy network, 3 reactions / 3 metabolites):
  - `T @ T.T ≈ Σ_r` (marginal dG covariance)
  - `s ≈ μ_r` (marginal dG mean)
  - Verify q ≤ n_rxn (rank-deficiency preserved)
  - Cross-check against `ThermoSamplingData.drg_mvn_params()` on the same input

### Marimo notebook: `notebooks/01_basis_construction.py`

Interactive demo: pick a small toy network, vary dfg0 uncertainties, visualise
the ellipsoid in 2D dG space.

---

## Step 2 — `ThermoSamplingData.from_model_data()` (no pta needed)

Add a classmethod that builds a `ThermoSamplingData` directly from
`ThermodynamicSpaceParams` + `FluxSpaceParams` using `pta_port.build_drg_basis`.
This is the zero-dependency construction path for new models.

```python
@classmethod
def from_model_data(
    cls,
    thermo: ThermodynamicSpaceParams,
    flux:   FluxSpaceParams,
    confidence_level: float = 0.95,
    min_drg: float = 1e-1,
) -> "ThermoSamplingData":
```

Test: construct `ThermoSamplingData` from toy parameters, compare
`drg_mvn_params()` output to the direct formula.

---

## Step 3 — Proper TFS sampler (Algorithm 1)

**This is the core contribution.**  Rewrite `thermo_sampler.py` to implement
Algorithm 1 faithfully in Python.

### Key differences from the current implementation

| Current `thermo_sampler.py` | Proper TFS |
|-----------------------------|------------|
| Works in dG space directly  | Works in basis m-space (isotropic) |
| Box constraint ±10σ         | Confidence sphere `\|\|m\|\|² ≤ r²` |
| Single-segment walk         | Multi-segment: enumerate all valid intervals |
| No feasibility check        | LP feasibility per orthant per interval |
| No orthant tracking         | Orthant counts → orthant probabilities |
| Not convergence-aware       | PSRF computation included |

### Mathematical specification

At each step, current point is `m ∈ R^q`.  Pick direction `d ~ Uniform(sphere)`.

**Step 3a — Sphere intersection** (closed form):

```
||m + t·d||² ≤ r²
||d||²·t² + 2·(m·d)·t + (||m||² - r²) ≤ 0
→ discriminant check, roots t_lo, t_hi
```

**Step 3b — Sign constraint intervals** (linear per reaction):

For forward-only reaction `i` (lb_F ≥ 0): need `(T[i,:]·(m+t·d) + s[i]) ≤ -ε`
For backward-only reaction `i` (ub_F ≤ 0): need `(T[i,:]·(m+t·d) + s[i]) ≥ +ε`

Each gives a half-line constraint on `t`.  Intersect all to get the feasible
interval for irreversible reactions: `[t_irrev_lo, t_irrev_hi]`.

Combined with sphere: `I_total = [max(t_lo, t_irrev_lo), min(t_hi, t_irrev_hi)]`.

**Step 3c — Orthant enumeration along the segment**:

For each reversible reaction `i`, the sign of `ΔrG'_i = T[i,:]·(m+t·d) + s[i]`
changes at most once along the segment (it's linear in t).  Find all sign-change
breakpoints within `I_total`.  This partitions `I_total` into at most `2^k` 
sub-intervals (in practice k is small per step — at most n+1 crossings for n
reversible reactions; use sorted crossing points).

**Step 3d — LP feasibility per sub-interval**:

For each sub-interval with orthant `o` (sign pattern of ΔrG' for reversible
reactions), check whether a steady-state flux solution exists:

```
Sv = 0,  lb ≤ v ≤ ub,  sign(v_i) = o_i  for reversible reactions i
```

Use `scipy.optimize.linprog` with a zero objective (feasibility only).
Cache results: `dict[(orthant_tuple)] → bool` — most orthants are revisited
many times across steps.

**Step 3e — CDF weighting**:

For each valid sub-interval `[a_k, b_k]`, compute the probability mass along
the 1-D marginal truncated normal on the ray (same as current code):

```python
t_mean = -dot(L_inv @ (m - μ_r), L_inv @ d) / dot(L_inv @ d, L_inv @ d)
# In basis space this simplifies to:
t_mean = -dot(m, d)  # because m ~ N(0,I), direction d on sphere
t_std  = 1.0
weight_k = Φ(b_k - t_mean) - Φ(a_k - t_mean)
```

Sample a segment proportional to weights, then sample a point within it from
the truncated normal.

**Step 3f — Orthant tracking**:

Maintain `orthant_counts: dict[tuple, int]`.  After sampling, increment the
count for the current orthant.

### Public API (backward compatible)

```python
@dataclass
class DRGSamplingResult:
    samples:       pd.DataFrame   # (n_total, n_rxn) — dG values
    basis_samples: pd.DataFrame   # (n_total, q)     — m values
    orthants:      pd.DataFrame   # orthant sign patterns + weight column


def find_initial_points(data, n, seed=None) -> np.ndarray:
    """(n_basis_dims, n) array of feasible m-space starting points."""

def sample_drg(
    data:            ThermoSamplingData,
    initial_points:  np.ndarray | None = None,
    num_samples:     int = 20_000,
    num_chains:      int = 4,
    num_initial_steps: int | None = None,
    max_psrf:        float = 1.05,
    seed:            int | None = None,
    feasibility_cache_size: int = 10_000,
    min_segment_length: float = 1e-6,
) -> DRGSamplingResult:
```

### Tests: `test_priors/test_tfs_sampler.py`

1. **Sphere intersection**: unit test `_sphere_bounds(m, d, r)` on known examples.
2. **Sign constraints**: `_irreversible_bounds(m, d, T, s, fwd_idx, rev_idx, eps)`.
3. **Orthant breakpoints**: `_orthant_breakpoints(m, d, T, s, rev_idx)` on a
   2-reaction reversible toy.
4. **LP feasibility**: test with a 2-metabolite / 2-reaction network, known
   orthants.
5. **Integration test** (`test_sample_drg_toy`):
   - 3-reaction network with 1 reversible reaction
   - After N=10_000 samples, orthant probabilities should match the analytical
     ratio (integral of the MVN over each half-space)
   - Check `DRGSamplingResult.samples` has correct sign for irreversible reactions
6. **PSRF** smoke test: 4 chains, `arviz.rhat < 1.1` on all dimensions.

### Marimo notebook: `notebooks/02_tfs_sampler.py`

- Step-by-step animation of Algorithm 1 on a 2D toy (visualise sphere +
  orthant crossings + LP check + segment weighting)
- Orthant probability bar chart vs. analytical benchmark

---

## Step 4 — Integration and `thermo_prior.py` cleanup

1. **`ThermoPrior._fill_caches()`**: currently ignores `DRGSamplingResult.orthants`
   (doesn't exist yet in the simplified result).  After Step 3, wire it up:
   use `orthants` DataFrame directly instead of computing sign patterns post-hoc
   from samples.

2. **`ThermoPrior.extract_drg_mvn()`**: works identically for both the pta
   `TFSModel` and the new `ThermoSamplingData` — no change needed.

3. **Remove** the `try: from pta.sampling.tfs import TFSModel` fallback once
   the new sampler is confirmed working.

### Tests: `test_priors/test_thermo.py` (extend existing)

- End-to-end test: `ThermoSamplingData.from_model_data(...)` → `sample_drg(...)` →
  `ThermoPrior._fill_caches(...)` on the toy iML1515-CAN-like fixture.
- Check that `ThermoPrior.log_prob()` runs without error on a handful of samples.

---

## Step 5 — Marimo presentation notebook

`notebooks/00_pta_port_overview.py` — high-level interactive demo:

1. Build `ThermodynamicSpaceParams` + `FluxSpaceParams` for a toy *E. coli*
   core network from scratch (no cobra, just hardcoded S / parameters)
2. Call `build_drg_basis()`, show T and ellipsoid
3. Run `find_initial_points()`, plot in 2D basis space
4. Run `sample_drg()`, show dG histograms and orthant pie chart
5. Show PSRF convergence trace

---

## Implementation order and dependencies

```
Step 1: pta_port/             ← no dependencies on other steps
   ↓
Step 2: ThermoSamplingData    ← requires Step 1
   ↓
Step 3: thermo_sampler.py     ← requires Step 1, Step 2 (for ThermoSamplingData)
   ↓
Step 4: thermo_prior.py       ← requires Step 3
   ↓
Step 5: notebooks             ← requires Step 3
```

---

## What is NOT ported (and why)

| pta component | Reason not ported |
|---------------|-------------------|
| `ThermodynamicSpace` (full) | Requires enkie/component_contribution for Gibbs estimates; raw estimates already in `ThermoSamplingData.dfg0_prime_*` fields |
| `PmoProblem` / PMO | Requires CVXPY + Gurobi; initial points found with `scipy.linprog` instead |
| `ConvergenceManager` | Handled inside `sample_drg()` with PSRF check |
| `sample_fluxes_from_drg()` | Already done in `ThermoPrior._fill_caches()` via `sample_polytope` |
| `FluxSpaceBasis` | Requires PolyRound; flux polytope sampling already in `sbmfi.core.polytopia` |
| C++ `pb.sample_free_energies` | Replaced by Algorithm 1 in pure Python |

---

## Open questions / risks

1. **LP feasibility cache performance**: the cache key is a tuple of `{-1, 0, +1}`
   signs for reversible reactions.  For large models (700 reversible reactions in
   iML1515-CAN), the cache may explode.  Mitigation: LRU cache with size limit;
   only reversible reactions in the orthant key.

2. **Orthant breakpoints complexity**: a segment crosses at most γ_rev breakpoints
   where γ_rev is the number of reversible reactions.  Sorting these is O(γ_rev log
   γ_rev) per step.  For large models this may dominate.  Mitigation: numba or
   vectorised numpy; profile first.

3. **Initial point quality**: `find_initial_points` currently works in dG space.
   After Step 3 it must return m-space points.  The linprog constraints need to be
   re-expressed in m-space: `T[i,:] · m ≤ -s[i] - ε` for forward reactions.

4. **`qr_rank_deficient` equivalence**: pta uses component_contribution's C++
   implementation.  Our scipy version must produce the same rank and span (up to
   rotation of the basis).  The test in Step 1 validates this numerically but
   differences may appear for near-degenerate cases.
