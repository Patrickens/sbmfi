from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbmfi.priors.thermo_data import ThermoSamplingData
from sbmfi.priors.thermo_sampler import find_initial_points, sample_drg


def _make_sampling_data():
    return ThermoSamplingData(
        T_reaction_ids=["R1", "R2", "R3"],
        metabolite_ids=["M1", "M2", "M3"],
        T_kelvin=310.15,
        dfg0_prime_mean=np.array([-2.0, 2.5, 0.0], dtype=float),
        log_conc_mean=np.zeros(3, dtype=float),
        dfg0_prime_cov_sqrt=np.diag([0.3, 0.2, 0.5]).astype(float),
        log_conc_cov=np.diag([1e-6, 1e-6, 1e-6]).astype(float),
        S_constraints=np.eye(3, dtype=float),
        F_reaction_ids=["R1", "R2", "R3"],
        F_lb=np.array([0.0, -10.0, -10.0], dtype=float),
        F_ub=np.array([10.0, 0.0, 10.0], dtype=float),
    )


def test_from_tfs_model_extracts_expected_fields():
    fake_tfs = SimpleNamespace(
        T=SimpleNamespace(
            reaction_ids=["R1", "R2"],
            S_constraints=pd.DataFrame(np.eye(2), index=["M1", "M2"], columns=["R1", "R2"]),
            parameters=SimpleNamespace(T=lambda: SimpleNamespace(model=310.15)),
            dfg0_prime_mean=SimpleNamespace(model=np.array([1.0, -1.0])),
            log_conc_mean=SimpleNamespace(model=np.array([0.1, -0.2])),
            _dfg0_prime_cov_sqrt=SimpleNamespace(model=np.eye(2)),
            log_conc_cov=SimpleNamespace(model=np.eye(2) * 0.5),
        ),
        F=SimpleNamespace(
            reaction_ids=["R1", "R2", "R3"],
            lb=np.array([0.0, -5.0, -1.0]),
            ub=np.array([5.0, 0.0, 1.0]),
        ),
    )

    data = ThermoSamplingData.from_tfs_model(fake_tfs)

    assert data.T_reaction_ids == ["R1", "R2"]
    assert data.metabolite_ids == ["M1", "M2"]
    assert data.F_reaction_ids == ["R1", "R2", "R3"]
    np.testing.assert_allclose(data.S_constraints, np.eye(2))


def test_find_initial_points_respects_irreversible_sign_constraints():
    data = _make_sampling_data()
    points = find_initial_points(data, n=6)

    assert points.shape == (3, 6)
    assert np.all(points[0] <= 0.0)
    assert np.all(points[1] >= 0.0)


def test_sample_drg_returns_expected_shapes_and_columns():
    data = _make_sampling_data()
    initial_points = find_initial_points(data, n=4)
    result = sample_drg(
        data,
        initial_points=initial_points,
        num_samples=60,
        num_chains=4,
        num_initial_steps=40,
        seed=7,
    )

    assert list(result.samples.columns) == data.T_reaction_ids
    assert list(result.basis_samples.columns) == data.T_reaction_ids
    assert result.samples.shape == (240, 3)
    assert result.basis_samples.shape == (240, 3)
    assert np.all(result.samples["R1"].to_numpy() <= 1e-9)
    assert np.all(result.samples["R2"].to_numpy() >= -1e-9)


def test_sample_drg_rhat_smoke_test():
    az = pytest.importorskip("arviz")
    data = _make_sampling_data()
    result = sample_drg(
        data,
        initial_points=find_initial_points(data, n=4),
        num_samples=80,
        num_chains=4,
        num_initial_steps=80,
        seed=11,
    )

    chains = 4
    draws = 80
    samples = result.samples.to_numpy().reshape(chains, draws, -1)
    idata = az.from_dict(posterior={"drg": samples})
    rhat = az.rhat(idata).to_array().values
    assert np.isfinite(rhat).all()
    assert np.nanmax(rhat) < 1.1
