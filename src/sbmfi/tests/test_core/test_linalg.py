import math
import numpy as np
import scipy.special
import pytest

# Import torch if available.
try:
    import torch
except ImportError:
    torch = None

from sbmfi.core.linalg import LinAlg

# -------------------------------------------------------------------
# Fixtures for instantiating the LinAlg object for each backend.
# -------------------------------------------------------------------
@pytest.fixture(params=["numpy", "torch"])
def linalg(request):
    """
    Fixture returning a LinAlg instance for each backend.
    """
    return LinAlg(backend=request.param, seed=123)

# -------------------------------------------------------------------
# Test get_tensor (creation with and without indices)
# -------------------------------------------------------------------
def test_get_tensor(linalg):
    # Test 1: Basic diagonal matrix creation
    shape = (3, 3)
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1, 2, 3])
    tensor = linalg.get_tensor(shape=shape, indices=indices, values=values,
                             dtype=np.int64, device=None)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(np.diag(tensor), [1, 2, 3])
    else:
        np.testing.assert_array_equal(tensor.diag().cpu().numpy(), np.array([1, 2, 3]))

    # Test 2: Dense tensor creation without indices
    dense_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor = linalg.get_tensor(shape=shape, indices=None, values=dense_values,
                             dtype=np.float64, device=None)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(tensor, dense_values)
    else:
        np.testing.assert_array_equal(tensor.cpu().numpy(), dense_values)

    # Test 3: 1D tensor creation
    shape_1d = (5,)
    values_1d = np.array([1, 2, 3, 4, 5])
    tensor = linalg.get_tensor(shape=shape_1d, indices=None, values=values_1d,
                             dtype=np.float32, device=None)
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(tensor, values_1d)
    else:
        np.testing.assert_array_equal(tensor.cpu().numpy(), values_1d)

    # Test 4: Sparse tensor with custom indices
    shape_sparse = (4, 4)
    indices_sparse = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    values_sparse = np.array([1.0, 2.0, 3.0, 4.0])
    tensor = linalg.get_tensor(shape=shape_sparse, indices=indices_sparse, values=values_sparse,
                             dtype=np.float64, device=None)
    expected = np.zeros(shape_sparse)
    for idx, val in zip(indices_sparse, values_sparse):
        expected[tuple(idx)] = val
    if linalg.backend == "numpy":
        np.testing.assert_array_equal(tensor, expected)
    else:
        np.testing.assert_array_equal(tensor.cpu().numpy(), expected)

    # Test 5: Empty tensor
    shape_empty = (0, 0)
    tensor = linalg.get_tensor(shape=shape_empty, indices=None, values=None,
                             dtype=np.float64, device=None)
    if linalg.backend == "numpy":
        assert tensor.shape == shape_empty
    else:
        assert tuple(tensor.shape) == shape_empty

    # Test 6: Tensor with different dtypes
    for dtype in [np.int32, np.float32, np.float64]:
        tensor = linalg.get_tensor(shape=(2, 2), indices=None,
                                 values=np.array([[1, 2], [3, 4]], dtype=dtype),
                                 dtype=dtype, device=None)
        if linalg.backend == "numpy":
            assert tensor.dtype == dtype
        else:
            assert tensor.dtype == torch.from_numpy(np.array(0, dtype=dtype)).dtype

    # Test 7: Tensor with 1D input
    tensor = linalg.get_tensor(shape=(1, 5), indices=None,
                             values=np.array([[1, 2, 3, 4, 5]]),
                             dtype=np.float64, device=None)
    if linalg.backend == "numpy":
        assert tensor.shape == (1, 5)
    else:
        assert tuple(tensor.shape) == (1, 5)

    # Test 8: Default dtype (dtype=None)
    # Test with integer values
    int_values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = linalg.get_tensor(shape=(2, 2), indices=None,
                             values=int_values, dtype=None, device=None)
    if linalg.backend == "numpy":
        assert tensor.dtype == int_values.dtype
    else:
        assert tensor.dtype == torch.from_numpy(int_values).dtype

    # Test with float values
    float_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor = linalg.get_tensor(shape=(2, 2), indices=None,
                             values=float_values, dtype=None, device=None)
    if linalg.backend == "numpy":
        assert tensor.dtype == float_values.dtype
    else:
        assert tensor.dtype == torch.from_numpy(float_values).dtype

# -------------------------------------------------------------------
# Test LU factorization and solving linear systems
# -------------------------------------------------------------------
def test_lu_solve(linalg):
    # Create numpy arrays for input
    A = np.array([[3, 1],
                  [1, 2]], dtype=np.float64)
    
    # Test 1: 1D b vector
    b_1d = np.array([9, 8], dtype=np.float64)
    A_tensor = linalg.get_tensor(values=A)
    b_tensor = linalg.get_tensor(values=b_1d)
    LU = linalg.LU(A_tensor)
    x = linalg.solve(LU, b_tensor)
    x_np = linalg.tonp(x)
    expected = np.linalg.solve(A, b_1d)
    np.testing.assert_allclose(x_np, expected, rtol=1e-5)
    
    # Test 2: 2D b vector (multiple right-hand sides)
    b_2d = np.array([[9, 8],
                     [7, 6]], dtype=np.float64)
    b_tensor = linalg.get_tensor(values=b_2d)
    x = linalg.solve(LU, b_tensor)
    x_np = linalg.tonp(x)
    expected = np.linalg.solve(A, b_2d)
    np.testing.assert_allclose(x_np, expected, rtol=1e-5)
    
    # Test 3: 3D A and b tensors (batch of different matrices)
    A_3d = np.array([[[3, 1],
                      [1, 2]],
                     [[2, 1],
                      [1, 3]]], dtype=np.float64)
    b_3d = np.array([[[9, 8],
                      [7, 6]],
                     [[5, 4],
                      [3, 2]]], dtype=np.float64)
    
    A_tensor = linalg.get_tensor(values=A_3d)
    b_tensor = linalg.get_tensor(values=b_3d)
    LU = linalg.LU(A_tensor)
    x = linalg.solve(LU, b_tensor)
    x_np = linalg.tonp(x)
    
    # For each matrix in the batch, solve separately
    expected = np.array([np.linalg.solve(A_3d[0], b_3d[0]),
                        np.linalg.solve(A_3d[1], b_3d[1])])
    
    np.testing.assert_allclose(x_np, expected, rtol=1e-5)

    # Test 4: Single A matrix (broadcasted to 3D) with multiple b values
    A_3d = A[None, ...]  # Shape: (1, 2, 2)
    
    A_tensor = linalg.get_tensor(values=A_3d)
    b_tensor = linalg.get_tensor(values=b_3d)
    LU = linalg.LU(A_tensor)
    x = linalg.solve(LU, b_tensor)
    x_np = linalg.tonp(x)
    
    # Expected result: solve each b matrix with the same A matrix
    expected = np.array([np.linalg.solve(A, b_3d[0]),
                        np.linalg.solve(A, b_3d[1])])
    
    np.testing.assert_allclose(x_np, expected, rtol=1e-5)

# -------------------------------------------------------------------
# Test convolution
# -------------------------------------------------------------------
def test_convolve(linalg):
    # Test 1: 1D array convolution
    a = np.array([1, 2, 3])
    v = np.array([0, 1])
    
    # Convert to backend tensors
    a_tensor = linalg.get_tensor(values=a)
    v_tensor = linalg.get_tensor(values=v)
    
    # Perform convolution
    conv_result = linalg.convolve(a_tensor, v_tensor)
    
    # Convert result to numpy for comparison
    conv_result_np = linalg.tonp(conv_result)
    expected = np.convolve(a, v)
    
    # Compare results
    np.testing.assert_allclose(conv_result_np, expected)

    # Test 2: 2D matrix convolution
    a_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])
    v_2d = np.array([[0, 1],
                     [1, 0]])
    
    # Convert to backend tensors
    a_2d_tensor = linalg.get_tensor(values=a_2d)
    v_2d_tensor = linalg.get_tensor(values=v_2d)
    
    # Perform convolution
    conv_result_2d = linalg.convolve(a_2d_tensor, v_2d_tensor)
    
    # Convert result to numpy for comparison
    conv_result_2d_np = linalg.tonp(conv_result_2d)
    
    # Expected result: convolve each row separately
    expected_2d = np.array([np.convolve(a_2d[0], v_2d[0]),
                           np.convolve(a_2d[1], v_2d[1])])
    
    # Compare results
    np.testing.assert_allclose(conv_result_2d_np, expected_2d)

# -------------------------------------------------------------------
# Test nonzero
# -------------------------------------------------------------------
def test_nonzero(linalg):
    A = np.array([[0, 1],
                  [2, 0]])
    indices, values = linalg.nonzero(A)
    # Expected nonzero indices (order might differ).
    expected_indices = [tuple(row) for row in np.array([[0, 1], [1, 0]])]
    res_indices = sorted([tuple(row) for row in indices])
    assert res_indices == sorted(expected_indices)
    # Expected nonzero values.
    expected_values = np.array([1, 2])
    np.testing.assert_array_equal(np.sort(values), np.sort(expected_values))

# -------------------------------------------------------------------
# Test tonp (conversion to NumPy array)
# -------------------------------------------------------------------
def test_tonp(linalg):
    if linalg.backend == "torch":
        t = linalg.get_tensor(shape=(2, 2), indices=None,
                              values=np.array([[1, 2], [3, 4]]),
                              dtype=np.float64)
        np_arr = linalg.tonp(t)
        np.testing.assert_array_equal(np_arr, np.array([[1, 2], [3, 4]]))
    else:
        t = np.array([[1, 2], [3, 4]])
        np_arr = linalg.tonp(t)
        np.testing.assert_array_equal(np_arr, t)


# -------------------------------------------------------------------
# Test random generation functions (randn, randu, randperm)
# -------------------------------------------------------------------
def test_rand_shapes(linalg):
    r1 = linalg.randn((4, 4))
    r2 = linalg.randu((4, 4))
    if linalg.backend == "numpy":
        assert r1.shape == (4, 4)
        assert r2.shape == (4, 4)
    else:
        assert tuple(r1.shape) == (4, 4)
        assert tuple(r2.shape) == (4, 4)

def test_randperm(linalg):
    perm = linalg.randperm(10)
    if linalg.backend == "numpy":
        assert set(perm) == set(range(10))
    else:
        assert set(perm.cpu().numpy()) == set(range(10))

# -------------------------------------------------------------------
# Test min and max (with return_indices)
# -------------------------------------------------------------------
def test_min_max(linalg):
    arr = np.array([[3, 1, 2],
                    [4, 0, 5]])
    min_val, min_idx = linalg.min(arr, dim=1, keepdims=False, return_indices=True)
    expected_min = np.argmin(arr, axis=1)
    np.testing.assert_array_equal(min_idx, expected_min)

    max_val, max_idx = linalg.max(arr, dim=1, keepdims=False, return_indices=True)
    expected_max = np.argmax(arr, axis=1)
    np.testing.assert_array_equal(max_idx, expected_max)

# -------------------------------------------------------------------
# Test logsumexp
# -------------------------------------------------------------------
def test_logsumexp(linalg):
    arr = np.array([1, 2, 3])
    result = linalg.logsumexp(arr, dim=0)
    expected = scipy.special.logsumexp(arr)
    if linalg.backend == "numpy":
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    else:
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

# -------------------------------------------------------------------
# Test basic tensor operations: permutax, transax, unsqueeze, cat.
# -------------------------------------------------------------------
def test_tensor_ops(linalg):
    if linalg.backend == "numpy":
        A = np.array([[1, 2],
                      [3, 4]])
        B = linalg.permutax(A, 1, 0)
        np.testing.assert_array_equal(B, A.T)
        C = linalg.transax(A, 0, 1)
        np.testing.assert_array_equal(C, A.T)
        D = linalg.unsqueeze(A, 0)
        np.testing.assert_array_equal(D, np.expand_dims(A, 0))
        E = linalg.cat([A, A], dim=0)
        np.testing.assert_array_equal(E, np.concatenate([A, A], axis=0))
    else:
        A = torch.tensor([[1, 2],
                          [3, 4]])
        B = linalg.permutax(A, 1, 0)
        np.testing.assert_array_equal(B.cpu().numpy(), A.t().cpu().numpy())
        C = linalg.transax(A, 0, 1)
        np.testing.assert_array_equal(C.cpu().numpy(), A.t().cpu().numpy())
        D = linalg.unsqueeze(A, 0)
        np.testing.assert_array_equal(D.cpu().numpy(), A.unsqueeze(0).cpu().numpy())
        E = linalg.cat([A, A], dim=0)
        np.testing.assert_array_equal(E.cpu().numpy(), torch.cat([A, A], dim=0).cpu().numpy())

# -------------------------------------------------------------------
# Test probability functions: norm_pdf, norm_log_pdf, norm_cdf, norm_inv_cdf.
# -------------------------------------------------------------------
def test_probability_functions(linalg):
    x = np.array([0.0, 0.5, 1.0])
    pdf = linalg.norm_pdf(x, mu=0.0, std=1.0)
    log_pdf = linalg.norm_log_pdf(x, mu=0.0, std=1.0)
    cdf = linalg.norm_cdf(x, mu=0.0, std=1.0)
    inv_cdf = linalg.norm_inv_cdf(cdf, mu=0.0, std=1.0)
    np.testing.assert_allclose(inv_cdf, x, rtol=1e-5)
    # Check that the log PDF equals log(pdf).
    np.testing.assert_allclose(log_pdf, np.log(pdf), rtol=1e-5) 