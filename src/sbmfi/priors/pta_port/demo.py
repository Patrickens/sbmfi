"""Marimo demo — PTA thermodynamic flux sampling with equilibrator_api.

Full pipeline, no pta / enkie / R / Gurobi:

    equilibrator_api  →  estimate_drg0_prime()
         ↓
    build_drg_basis_from_reactions()  →  T, s
         ↓
    ThermoSamplingData  →  sample_drg()  (TFS Algorithm 1)
         ↓
    DRGSamplingResult  (dG' samples + orthant probabilities)

Run with:
    marimo edit src/sbmfi/priors/pta_port/demo.py
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    _src = Path(__file__).parent.parent.parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats.distributions import chi2
    from scipy.stats import multivariate_normal as MVN

    from sbmfi.priors.pta_port import (
        make_component_contribution,
        estimate_drg0_prime,
        build_drg_basis_from_reactions,
    )
    from sbmfi.priors.thermo_data import ThermoSamplingData
    from sbmfi.priors.thermo_sampler import find_initial_points, sample_drg
    return (
        MVN, Path, ThermoSamplingData, build_drg_basis_from_reactions,
        chi2, estimate_drg0_prime, find_initial_points, make_component_contribution,
        matplotlib, mo, np, plt, sample_drg, sys,
    )


# ─────────────────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    # Thermodynamic Flux Sampling — from eQuilibrator to dG' samples

    **Paper**: Gollub, Kaltenbach & Stelling (2020), doi:10.1101/2020.08.14.250845

    **Stack** (no pta / enkie / R / Gurobi required):

    | Step | Tool |
    |---|---|
    | Standard Gibbs energies + covariance | `equilibrator_api.ComponentContribution` |
    | Basis construction: $\Delta_rG' = T\mathbf{m}+\mathbf{s}$ | `pta_port.build_drg_basis_from_reactions` |
    | Hit-and-Run sampling (Algorithm 1) | `pta_port` TFS sampler |
    """)
    return


# ─────────────────────────────────────────────────────────
#  § 1 — The Model
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The Model

    The Gibbs free energy of reaction $i$ is (Eq. 2):

    $$\Delta_r G'_i = \underbrace{\Delta_r G'^{\circ}_i}_{\text{eQuilibrator}}
    + RT\,S_i^\top \ln\mathbf{c}$$

    The joint uncertainty in $\Delta_r G'^{\circ}$ and $\ln\mathbf{c}$ is MVN.
    A rank-revealing SVD reduces the covariance to a minimal basis of dimension $q$:

    $$\Delta_r\mathbf{G}' = T\mathbf{m}+\mathbf{s},\quad
    \mathbf{m}\sim\mathcal{N}(0,I_q),\quad TT^\top=\Sigma_r$$

    The **confidence sphere** in $\mathbf{m}$-space ($\|\mathbf{m}\|^2\le r^2$) plus the
    **second law** ($v_i\cdot\Delta_rG'_i<0$) and **steady-state LP** together define
    the feasible orthants.  TFS samples from this intersection using Hit-and-Run.
    """)
    return


# ─────────────────────────────────────────────────────────
#  § 2 — eQuilibrator
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Getting Thermodynamic Data from eQuilibrator

    We use two E. coli glycolysis **isomerases** — reactions with no cofactors, so
    the formation energies translate directly to reaction energies:

    | Reaction | Formula | KEGG |
    |---|---|---|
    | **PGI** — phosphoglucose isomerase | G6P ⇌ F6P | `kegg:C00092 = kegg:C00085` |
    | **TPI** — triosephosphate isomerase | DHAP ⇌ G3P | `kegg:C00111 = kegg:C00118` |

    Both are **reversible** in the model.  The eQuilibrator values are at
    E. coli cytoplasmic conditions (pH 7.8, ionic strength 0.1 M, T = 310.15 K).
    """)
    return


@app.cell
def _(mo):
    s_pH  = mo.ui.slider(6.5, 8.5, value=7.8, step=0.1,  label="pH (cytoplasm)")
    s_I   = mo.ui.slider(0.05, 0.5, value=0.1, step=0.05, label="ionic strength I (M)")
    s_T   = mo.ui.slider(298.15, 315.0, value=310.15, step=0.5, label="T (K)")
    s_lc  = mo.ui.slider(0.1, 1.5, value=0.5, step=0.05, label="σ(ln c) — conc. uncertainty")
    s_a   = mo.ui.slider(0.80, 0.999, value=0.95, step=0.005, label="α — confidence level")
    mo.vstack([mo.hstack([s_pH, s_I, s_T]), mo.hstack([s_lc, s_a])])
    return s_I, s_T, s_a, s_lc, s_pH


@app.cell
def _(
    ThermoSamplingData, build_drg_basis_from_reactions,
    chi2, estimate_drg0_prime, make_component_contribution,
    mo, np, s_I, s_T, s_a, s_lc, s_pH,
):
    _cc = make_component_contribution()

    _formulas = [
        "kegg:C00092 = kegg:C00085",   # PGI: G6P -> F6P
        "kegg:C00111 = kegg:C00118",   # TPI: DHAP -> G3P
    ]
    _drg0_mean, _drg0_cov_sqrt = estimate_drg0_prime(
        _formulas,
        pH=s_pH.value, ionic_strength_M=s_I.value, T_kelvin=s_T.value,
        cc=_cc,
    )

    # Stoichiometry sub-matrix for the 4 thermo-constrained metabolites
    # mets: g6p, f6p, dhap, g3p   rxns: PGI, TPI
    _S_con = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    _lc_cov = np.diag([s_lc.value**2] * 4)

    _T_mat, _s_vec = build_drg_basis_from_reactions(
        _drg0_mean, _drg0_cov_sqrt,
        log_conc_mean=np.zeros(4),
        log_conc_cov=_lc_cov,
        S_constraints=_S_con,
        T_kelvin=s_T.value,
    )
    _q = _T_mat.shape[1]
    _r = float(np.sqrt(chi2.ppf(s_a.value, _q)))
    _Sig_r = _T_mat @ _T_mat.T

    # Compute reachable dG range for each reaction within the sphere
    _sig_pgi = float(np.sqrt(_Sig_r[0, 0]))
    _sig_tpi = float(np.sqrt(_Sig_r[1, 1]))

    # Full flux network: bidirectional exchanges allow each pathway to run either way
    _flux_S = np.array([
        [ 1., -1.,  0.,  0.,  0.,  0.],  # g6p
        [ 0.,  1., -1.,  0.,  0.,  0.],  # f6p
        [ 0.,  0.,  0.,  1., -1.,  0.],  # dhap
        [ 0.,  0.,  0.,  0.,  1., -1.],  # g3p
    ])
    _F_lb = np.full(6, -10.0)   # bidirectional — any metabolite can be imported or exported
    _F_ub = np.full(6,  10.0)

    _sdata = ThermoSamplingData(
        T_reaction_ids=["PGI", "TPI"],
        metabolite_ids=["g6p", "f6p", "dhap", "g3p"],
        T_kelvin=s_T.value,
        dfg0_prime_mean=np.zeros(4),
        dfg0_prime_cov_sqrt=np.eye(4),
        log_conc_mean=np.zeros(4),
        log_conc_cov=_lc_cov,
        S_constraints=_S_con,
        flux_S=_flux_S,
        F_reaction_ids=["EX_g6p", "PGI", "EX_f6p", "EX_dhap", "TPI", "EX_g3p"],
        F_lb=_F_lb, F_ub=_F_ub,
        basis_to_drg_T=_T_mat,
        basis_to_drg_shift=_s_vec.reshape(-1, 1),
        confidence_radius=_r,
        drg_epsilon=0.1,
    )

    mo.md(f"""
    ### eQuilibrator results  (pH {s_pH.value}, I {s_I.value} M, T {s_T.value} K)

    | | PGI (G6P ⇌ F6P) | TPI (DHAP ⇌ G3P) |
    |---|---|---|
    | **ΔrG'°** (kJ/mol) | **{_drg0_mean[0]:.2f}** | **{_drg0_mean[1]:.2f}** |
    | σ (kJ/mol) | {_sig_pgi:.2f} | {_sig_tpi:.2f} |
    | dG range in sphere | [{_s_vec[0]-_r*_sig_pgi:.1f}, {_s_vec[0]+_r*_sig_pgi:.1f}] | [{_s_vec[1]-_r*_sig_tpi:.1f}, {_s_vec[1]+_r*_sig_tpi:.1f}] |

    > **Key insight**: at pH 7.8 with equal metabolite concentrations (log_conc_mean = 0),
    > both reactions have positive ΔrG'° — meaning they are thermodynamically *unfavourable*
    > in the forward direction.  In vivo, metabolite concentration ratios (G6P >> F6P,
    > DHAP >> G3P) drive them forward.  Increasing **σ(ln c)** widens the distribution
    > and allows the forward direction to appear.
    """)
    return _F_lb, _F_ub, _S_con, _Sig_r, _T_mat, _cc, _drg0_cov_sqrt, _drg0_mean, _flux_S, _lc_cov, _q, _r, _s_vec, _sdata, _sig_pgi, _sig_tpi


# ─────────────────────────────────────────────────────────
#  § 3 — Confidence region
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Confidence Region

    The sphere in **m-space** (isotropic) maps to a correlated ellipse in **dG-space**.
    The vertical axis (TPI) may not cross zero — when it does not, TPI is thermodynamically
    locked to the backward direction under the specified conditions.
    """)
    return


@app.cell
def _(_Sig_r, _T_mat, _r, _s_vec, mo, np, plt, s_a):
    _th = np.linspace(0, 2*np.pi, 500)
    _dg_ring = _T_mat @ (_r * np.vstack([np.cos(_th), np.sin(_th)])) + _s_vec[:, None]

    _g = max(abs(_s_vec).max() + 3*np.sqrt(_Sig_r.diagonal()).max(), _r*np.linalg.norm(_T_mat)*1.3)

    _fig1, (_axm, _axdg) = plt.subplots(1, 2, figsize=(11, 5))

    # m-space: sphere
    _axm.plot(*(_r*np.vstack([np.cos(_th), np.sin(_th)])), "C0", lw=2)
    _axm.scatter(0, 0, s=80, color="C1", zorder=6, label="origin (mean)")
    _axm.set_xlim(-_r*1.5, _r*1.5); _axm.set_ylim(-_r*1.5, _r*1.5)
    _axm.set_aspect("equal"); _axm.legend(fontsize=8)
    _axm.set_xlabel("$m_1$"); _axm.set_ylabel("$m_2$")
    _axm.set_title(f"m-space  (sphere, r={_r:.2f})")

    # dG-space: ellipse
    _axdg.plot(_dg_ring[0], _dg_ring[1], "C0", lw=2,
               label=f"{int(s_a.value*100)}% confidence ellipse")
    _axdg.axvline(0, color="r",      lw=1.2, ls="--", alpha=0.7, label=r"$\Delta_rG'_{PGI}=0$")
    _axdg.axhline(0, color="orange", lw=1.2, ls="--", alpha=0.7, label=r"$\Delta_rG'_{TPI}=0$")
    _axdg.scatter(*_s_vec, s=90, color="C1", zorder=6, label="mean s")
    _axdg.set_xlim(_s_vec[0]-_g, _s_vec[0]+_g)
    _axdg.set_ylim(_s_vec[1]-_g, _s_vec[1]+_g)
    _axdg.set_aspect("equal"); _axdg.legend(fontsize=8)
    _axdg.set_xlabel(r"$\Delta_rG'_{PGI}$ (kJ/mol)")
    _axdg.set_ylabel(r"$\Delta_rG'_{TPI}$ (kJ/mol)")
    _axdg.set_title("dG-space  (correlated ellipse)")

    plt.suptitle(
        "If the ellipse does not cross the zero line for TPI, "
        "TPI is locked to backward under these conditions\n"
        "(increase σ(ln c) to see the ellipse grow)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    mo.pyplot(_fig1)
    return


# ─────────────────────────────────────────────────────────
#  § 4 — Algorithm
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Hit-and-Run Algorithm (Algorithm 1)

    | Step | Action |
    |---|---|
    | **1** | Random direction $\mathbf{d}$ on $\mathbb{S}^{q-1}$ |
    | **2** | Sphere intersection $\to [\tau_-, \tau_+]$ |
    | **3** | Sign-change breakpoints $\tau_i^* = -\Delta_rG'_i(\mathbf{m})/(T_i^\top\mathbf{d})$ |
    | **4** | Per-segment LP: $\exists\,\mathbf{v}$ s.t. $S\mathbf{v}=0$, bounds, sign constraints? |
    | **5** | Weight valid segments by CDF mass of $\mathcal{N}(-\mathbf{m}{\cdot}\mathbf{d},1)$ |
    | **6** | Sample $\tau \to$ update $\mathbf{m}$ |

    The **LP feasibility check** (Step 4) uses the full flux network.  With
    bidirectional exchange reactions, any sign pattern is potentially feasible as
    long as a valid steady-state flux exists.
    """)
    return


# ─────────────────────────────────────────────────────────
#  § 5 — dG' Samples
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("## 5 · dG' Samples from TFS")
    return


@app.cell
def _(mo):
    _s_ns = mo.ui.slider(200, 3000, value=1000, step=100, label="samples per chain")
    _s_nc = mo.ui.slider(2, 8,    value=4,    step=1,   label="chains")
    _s_bi = mo.ui.slider(100, 2000, value=800, step=100, label="burn-in steps")
    mo.hstack([_s_ns, _s_nc, _s_bi])
    return _s_bi, _s_nc, _s_ns


@app.cell
def _(_s_bi, _s_nc, _s_ns, _sdata, find_initial_points, mo, sample_drg):
    _init = find_initial_points(_sdata, n=_s_nc.value)
    _res  = sample_drg(
        _sdata,
        initial_points=_init,
        num_samples=_s_ns.value,
        num_chains=_s_nc.value,
        num_initial_steps=_s_bi.value,
        seed=42,
    )
    _samp = _res.samples.to_numpy()

    mo.md(f"""
    **Done** — {len(_samp):,} samples  ({_s_nc.value} chains × {_s_ns.value})

    | | Mean ΔrG' | Std | P(forward) = P(< 0) |
    |---|---|---|---|
    | PGI | {_samp[:,0].mean():.2f} kJ/mol | {_samp[:,0].std():.2f} | {(_samp[:,0]<0).mean():.3f} |
    | TPI | {_samp[:,1].mean():.2f} kJ/mol | {_samp[:,1].std():.2f} | {(_samp[:,1]<0).mean():.3f} |

    **Interpretation**: At pH {round(_sdata.T_kelvin*0, 1)} with equal metabolite concentrations,
    P(PGI forward) = {(_samp[:,0]<0).mean():.1%} and P(TPI forward) = {(_samp[:,1]<0).mean():.1%}.
    In E. coli glycolysis, [G6P] >> [F6P] and [DHAP] >> [G3P] drive both reactions
    forward — increase **σ(ln c)** to see this effect.
    """)
    return _init, _res, _samp


# ── 5a. 2D scatter ────────────────────────────────────────

@app.cell
def _(_Sig_r, _dg_ring, _res, _s_nc, _s_ns, _s_vec, _samp, mo, np, plt):
    from scipy.stats import multivariate_normal as _MVN

    _nc, _ns = _s_nc.value, _s_ns.value
    _fig3, (_aL, _aR) = plt.subplots(1, 2, figsize=(12, 5.5))

    _gx = np.linspace(_samp[:,0].min()-1, _samp[:,0].max()+1, 200)
    _gy = np.linspace(_samp[:,1].min()-1, _samp[:,1].max()+1, 200)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _Z  = _MVN(_s_vec, _Sig_r).pdf(np.stack([_GX.ravel(), _GY.ravel()], 1)).reshape(200, 200)

    _aL.contourf(_GX, _GY, _Z, levels=8, cmap="Blues", alpha=0.35)
    for _c in range(_nc):
        _sl = slice(_c*_ns, (_c+1)*_ns)
        _aL.scatter(_samp[_sl,0], _samp[_sl,1], s=3, alpha=0.3,
                    rasterized=True, label=f"chain {_c+1}")
    _aL.plot(_dg_ring[0], _dg_ring[1], "k", lw=1.5, alpha=0.6, label="conf. ellipse")
    _aL.axvline(0, color="r",      lw=1, ls="--", alpha=0.6)
    _aL.axhline(0, color="orange", lw=1, ls="--", alpha=0.6)
    _aL.scatter(*_s_vec, s=100, color="C1", zorder=6, label="mean s")
    _aL.set_xlabel(r"$\Delta_rG'_{PGI}$ (kJ/mol)"); _aL.set_ylabel(r"$\Delta_rG'_{TPI}$ (kJ/mol)")
    _aL.set_title("Samples by chain"); _aL.legend(fontsize=7, markerscale=4)

    _hb = _aR.hexbin(_samp[:,0], _samp[:,1], gridsize=35, cmap="YlOrRd", mincnt=1, bins="log")
    _aR.plot(_dg_ring[0], _dg_ring[1], "C0", lw=1.5, alpha=0.7)
    _aR.axvline(0, color="r",      lw=1, ls="--", alpha=0.6)
    _aR.axhline(0, color="orange", lw=1, ls="--", alpha=0.6)
    _aR.scatter(*_s_vec, s=100, color="C1", zorder=6)
    plt.colorbar(_hb, ax=_aR, label="log(count)")
    _aR.set_xlabel(r"$\Delta_rG'_{PGI}$ (kJ/mol)"); _aR.set_ylabel(r"$\Delta_rG'_{TPI}$ (kJ/mol)")
    _aR.set_title("Joint density (log hexbin)")

    plt.suptitle(
        r"Sampled $\mathcal{T}$ — E. coli isomerases at cytoplasmic pH (real eQuilibrator data)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    mo.pyplot(_fig3)
    return


# ── 5b. Marginals ─────────────────────────────────────────

@app.cell
def _(_s_vec, _samp, mo, np, plt):
    from scipy.stats import norm as _N

    _fig4, _axs = plt.subplots(1, 2, figsize=(11, 4))
    for _j, (_nm, _col) in enumerate([("PGI", "C0"), ("TPI", "C1")]):
        _ax = _axs[_j]
        _ax.hist(_samp[:, _j], bins=60, density=True, color=_col, alpha=0.65, label="TFS samples")
        _xs = np.linspace(_samp[:, _j].min()-0.5, _samp[:, _j].max()+0.5, 400)
        _ax.plot(_xs, _N.pdf(_xs, _s_vec[_j], _samp[:, _j].std()), "k--", lw=1.5,
                 label=f"mean={_s_vec[_j]:.1f} kJ/mol")
        _ax.axvline(0, color="r", lw=1.2, ls="--", alpha=0.7, label="ΔrG'=0")
        _ax.set_xlabel(fr"$\Delta_rG'_{{\rm {_nm}}}$ (kJ/mol)", fontsize=11)
        _ax.set_ylabel("density"); _ax.set_title(_nm, fontsize=11); _ax.legend(fontsize=8)

    plt.suptitle(
        "Marginal distributions — real eQuilibrator values, equal-concentration prior\n"
        "Increase σ(ln c) to broaden; use realistic concentration data to shift means",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    mo.pyplot(_fig4)
    return


# ── 5c. Convergence + orthants ────────────────────────────

@app.cell
def _(_res, _s_nc, _s_ns, _samp, mo):
    az  = __import__("arviz", fromlist=["rhat", "from_dict"])
    _nc = _s_nc.value; _ns = _s_ns.value
    _rh = az.rhat(az.from_dict(posterior={
        "PGI": _samp.reshape(_nc, _ns, 2)[:, :, 0],
        "TPI": _samp.reshape(_nc, _ns, 2)[:, :, 1],
    }))
    _psrf = _res.psrf.values

    _orth_md = _res.orthants.to_markdown(index=False) if len(_res.orthants) else "_no orthants_"

    mo.md(f"""
    ### Convergence  ($\\hat R < 1.1$ = converged)

    | | arviz $\\hat R$ | internal PSRF | |
    |---|---|---|---|
    | PGI | {float(_rh["PGI"].values):.4f} | {_psrf[0]:.4f} | {"✓" if float(_rh["PGI"].values)<1.1 else "✗"} |
    | TPI | {float(_rh["TPI"].values):.4f} | {_psrf[1]:.4f} | {"✓" if float(_rh["TPI"].values)<1.1 else "✗"} |

    ### Orthant probabilities (from LP-feasibility tracking)

    {_orth_md}

    PGI = −1 means ΔrG'_PGI < 0 (forward direction); +1 means backward.
    """)
    return


# ─────────────────────────────────────────────────────────
#  § 6 — How to use with your own model
# ─────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Using with Your Own E. coli Model

    ```python
    from sbmfi.priors.pta_port import (
        make_component_contribution,
        estimate_drg0_prime,
        build_drg_basis_from_reactions,
    )
    from sbmfi.priors.thermo_data import ThermoSamplingData
    from sbmfi.priors.thermo_sampler import find_initial_points, sample_drg

    # 1. Build eQuilibrator (uses local cache — no network needed)
    cc = make_component_contribution()

    # 2. Define reactions as equilibrator_api formula strings
    formulas = [
        "kegg:C00092 = kegg:C00085",   # PGI
        "kegg:C00597 = kegg:C00631",   # PGM  (3-PG <-> 2-PG)
        # ... one string per thermodynamically-constrained reaction
    ]
    drg0_mean, drg0_cov_sqrt = estimate_drg0_prime(
        formulas, pH=7.8, ionic_strength_M=0.1, T_kelvin=310.15, cc=cc,
    )

    # 3. Build the reduced basis  ΔrG' = T @ m + s
    T_mat, s_vec = build_drg_basis_from_reactions(
        drg0_mean, drg0_cov_sqrt,
        log_conc_mean=...,   # (n_met,) from metabolomics or physiological prior
        log_conc_cov=...,    # (n_met, n_met)
        S_constraints=...,   # (n_met, n_rxn) stoichiometric sub-matrix
        T_kelvin=310.15,
    )

    # 4. Package into ThermoSamplingData and sample
    sdata = ThermoSamplingData(
        T_reaction_ids=reaction_ids, ...,
        basis_to_drg_T=T_mat, basis_to_drg_shift=s_vec.reshape(-1,1), ...
    )
    result = sample_drg(sdata, num_samples=5000, num_chains=4, seed=0)
    ```

    **Compound ID formats** accepted by equilibrator_api:

    | Format | Example |
    |---|---|
    | KEGG | `kegg:C00092` |
    | BiGG metabolite | `bigg.metabolite:g6p` |
    | ChEBI | `chebi:CHEBI:17665` |
    | MetaNetX | `metanetx.chemical:MNXM182` |
    """)
    return


if __name__ == "__main__":
    app.run()
