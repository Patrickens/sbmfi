"""
Tests for sbmfi.core.util — get_tensor, convolve, min_pos_max_neg,
with_generator, and the regex helpers.
"""
import pytest
import numpy as np
import torch
from sbmfi.core.util import (
    get_tensor,
    convolve,
    min_pos_max_neg,
    with_generator,
    tensormul_T,
    scale,
    sample_unit_hyper_sphere_ball,
    make_multidex,
)
import pandas as pd

DTYPE = torch.double


# ---------------------------------------------------------------------------
# get_tensor
# ---------------------------------------------------------------------------

class TestGetTensor:
    def test_zeros_from_shape(self):
        t = get_tensor(shape=(3, 4))
        assert t.shape == (3, 4)
        assert (t == 0).all()

    def test_dtype_float64_default(self):
        t = get_tensor(shape=(2, 2))
        assert t.dtype == torch.double

    def test_from_values_no_shape(self):
        vals = np.array([1.0, 2.0, 3.0])
        t = get_tensor(values=vals)
        torch.testing.assert_close(t, torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE))

    def test_sparse_scatter(self):
        # 3×3 tensor with two entries
        indices = np.array([[0, 1], [2, 2]])
        values = np.array([5.0, 7.0])
        t = get_tensor(shape=(3, 3), indices=indices, values=values)
        assert t[0, 1].item() == pytest.approx(5.0)
        assert t[2, 2].item() == pytest.approx(7.0)
        assert t[0, 0].item() == 0.0

    def test_duplicate_indices_summed(self):
        indices = np.array([[0, 0], [0, 0]])
        values = np.array([3.0, 4.0])
        t = get_tensor(shape=(2, 2), indices=indices, values=values)
        assert t[0, 0].item() == pytest.approx(7.0)

    def test_bool_dtype(self):
        t = get_tensor(shape=(3,), dtype=np.bool_)
        assert t.dtype == torch.bool

    def test_int32_dtype(self):
        t = get_tensor(shape=(4,), dtype=np.int32)
        assert t.dtype == torch.int32

    def test_fill_shape_from_values(self):
        """shape given but no indices → fill entire tensor with scalar value."""
        t = get_tensor(shape=(2, 3), values=np.array(9.0))
        assert (t == 9.0).all()

    def test_empty_indices(self):
        """Empty indices array should leave tensor all-zeros."""
        indices = np.empty((0, 2), dtype=np.int64)
        values = np.empty(0)
        t = get_tensor(shape=(3, 3), indices=indices, values=values)
        assert (t == 0).all()


# ---------------------------------------------------------------------------
# convolve
# ---------------------------------------------------------------------------

class TestConvolve:
    def test_1d_matches_numpy(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
        v = torch.tensor([0.0, 1.0, 0.5], dtype=DTYPE)
        expected = torch.tensor(np.convolve(a.numpy(), v.numpy()), dtype=DTYPE)
        result = convolve(a, v)
        torch.testing.assert_close(result, expected, atol=1e-9, rtol=0)

    def test_1d_output_length(self):
        a = torch.randn(5, dtype=DTYPE)
        v = torch.randn(3, dtype=DTYPE)
        result = convolve(a, v)
        assert result.shape == (5 + 3 - 1,)

    def test_2d_batch_shape(self):
        B, La, Lv = 4, 6, 3
        a = torch.randn(B, La, dtype=DTYPE)
        v = torch.randn(B, Lv, dtype=DTYPE)
        result = convolve(a, v)
        assert result.shape == (B, La + Lv - 1)

    def test_2d_matches_1d_per_row(self):
        B = 3
        a = torch.randn(B, 5, dtype=DTYPE)
        v = torch.randn(B, 4, dtype=DTYPE)
        batch = convolve(a, v)
        for i in range(B):
            row = convolve(a[i], v[i])
            torch.testing.assert_close(batch[i], row, atol=1e-9, rtol=0)

    def test_invalid_ndim_raises(self):
        a = torch.randn(2, 3, 4, dtype=DTYPE)
        v = torch.randn(2, 3, 4, dtype=DTYPE)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            convolve(a, v)


# ---------------------------------------------------------------------------
# min_pos_max_neg
# ---------------------------------------------------------------------------

class TestMinPosMaxNeg:
    @pytest.fixture
    def alpha(self):
        return torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=DTYPE)

    def test_returns_both_by_default(self, alpha):
        alpha_min, alpha_max = min_pos_max_neg(alpha)
        assert alpha_min.item() == pytest.approx(-1.0)   # max negative
        assert alpha_max.item() == pytest.approx(2.0)    # min positive

    def test_return_what_1_only_max(self, alpha):
        result = min_pos_max_neg(alpha, return_what=1)
        assert result.item() == pytest.approx(2.0)

    def test_return_what_minus1_only_min(self, alpha):
        result = min_pos_max_neg(alpha, return_what=-1)
        assert result.item() == pytest.approx(-1.0)

    def test_keepdims(self, alpha):
        alpha_min, alpha_max = min_pos_max_neg(alpha.unsqueeze(0), keepdims=True)
        assert alpha_min.shape == (1, 1)
        assert alpha_max.shape == (1, 1)

    def test_return_indices(self, alpha):
        (alpha_min, idx_min), (alpha_max, idx_max) = min_pos_max_neg(
            alpha, return_indices=True
        )
        assert alpha[idx_min].item() == pytest.approx(-1.0)
        assert alpha[idx_max].item() == pytest.approx(2.0)

    def test_batch_2d(self):
        alpha = torch.tensor([
            [-2.0,  3.0, -0.5,  1.0],
            [-4.0, -1.0,  0.5,  2.0],
        ], dtype=DTYPE)
        alpha_min, alpha_max = min_pos_max_neg(alpha)
        torch.testing.assert_close(alpha_min, torch.tensor([-0.5, -1.0], dtype=DTYPE))
        torch.testing.assert_close(alpha_max, torch.tensor([1.0, 0.5], dtype=DTYPE))


# ---------------------------------------------------------------------------
# with_generator decorator
# ---------------------------------------------------------------------------

class TestWithGenerator:
    def test_injects_generator_when_none(self):
        @with_generator
        def sample(n, generator):
            return torch.rand(n, generator=generator)

        result = sample(5)
        assert result.shape == (5,)

    def test_uses_passed_generator(self):
        @with_generator
        def sample(n, generator):
            return torch.rand(n, generator=generator)

        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        r1 = sample(10, generator=g1)
        r2 = sample(10, generator=g2)
        torch.testing.assert_close(r1, r2)

    def test_preserves_function_name(self):
        @with_generator
        def my_func(x, generator):
            return x

        assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# make_multidex
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tensormul_T
# ---------------------------------------------------------------------------

class TestTensormulT:
    def test_2d(self):
        A = torch.randn(3, 4, dtype=DTYPE)
        x = torch.randn(5, 4, dtype=DTYPE)
        result = tensormul_T(A, x)
        expected = (A @ x.T).T
        torch.testing.assert_close(result, expected)

    def test_batched(self):
        A = torch.randn(7, 3, 4, dtype=DTYPE)
        x = torch.randn(7, 5, 4, dtype=DTYPE)
        result = tensormul_T(A, x)
        expected = torch.transpose(A @ torch.transpose(x, -2, -1), -2, -1)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_forward(self):
        lo = torch.tensor(2.0, dtype=DTYPE)
        hi = torch.tensor(6.0, dtype=DTYPE)
        x = torch.tensor(4.0, dtype=DTYPE)
        assert scale(x, lo, hi).item() == pytest.approx(0.5)

    def test_reverse(self):
        lo = torch.tensor(2.0, dtype=DTYPE)
        hi = torch.tensor(6.0, dtype=DTYPE)
        u = torch.tensor(0.5, dtype=DTYPE)
        assert scale(u, lo, hi, rev=True).item() == pytest.approx(4.0)

    def test_roundtrip(self):
        lo = torch.zeros(4, dtype=DTYPE)
        hi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
        x = torch.rand(4, dtype=DTYPE) * hi
        u = scale(x, lo, hi)
        x_back = scale(u, lo, hi, rev=True)
        torch.testing.assert_close(x_back, x, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# sample_unit_hyper_sphere_ball
# ---------------------------------------------------------------------------

class TestSampleUnitHyperSphereBall:
    def test_sphere_unit_norm(self):
        samples = sample_unit_hyper_sphere_ball((100, 5))
        norms = torch.norm(samples, dim=-1)
        torch.testing.assert_close(norms, torch.ones(100, dtype=DTYPE), atol=1e-6, rtol=0)

    def test_ball_inside_unit_sphere(self):
        samples = sample_unit_hyper_sphere_ball((200, 4), ball=True)
        norms = torch.norm(samples, dim=-1)
        assert (norms <= 1.0 + 1e-6).all()

    def test_hemi_positive_first_coord(self):
        samples = sample_unit_hyper_sphere_ball((100, 3), hemi=True)
        assert (samples[..., 0] >= 0).all()

    def test_reproducible(self):
        g1 = torch.Generator().manual_seed(77)
        g2 = torch.Generator().manual_seed(77)
        s1 = sample_unit_hyper_sphere_ball((10, 3), generator=g1)
        s2 = sample_unit_hyper_sphere_ball((10, 3), generator=g2)
        torch.testing.assert_close(s1, s2)


# ---------------------------------------------------------------------------
# make_multidex
# ---------------------------------------------------------------------------

class TestMakeMultidex:
    def test_basic(self):
        idx = pd.Index(["x", "y", "z"], name="metabolite")
        result = make_multidex({"exp1": idx, "exp2": idx})
        assert isinstance(result, pd.MultiIndex)
        assert result.names[0] == "labelling_id"
        assert result.names[1] == "metabolite"
        assert len(result) == 6

    def test_custom_name0(self):
        idx = pd.Index(["a", "b"], name="item")
        result = make_multidex({"L1": idx}, name0="label")
        assert result.names[0] == "label"
