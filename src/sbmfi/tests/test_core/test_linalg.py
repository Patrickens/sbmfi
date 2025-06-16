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
    A_tensor = linalg.get_tensor(values=A)
    indices, values = linalg.nonzero(A_tensor)
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
    A = np.array([[
            [3, 1, 2],
            [4, 0, 5]
        ],[
            [3, 1, 2],
            [4, 0, 5]
        ]])
    A_tensor = linalg.get_tensor(values=A)

    # Test 1: Basic min/max along dimension
    min_val, min_idx = linalg.min(A_tensor, dim=1, keepdims=False, return_indices=True)
    expected_min = np.argmin(A, axis=1)
    np.testing.assert_array_equal(min_idx, expected_min)

    max_val, max_idx = linalg.max(A_tensor, dim=1, keepdims=False, return_indices=True)
    expected_max = np.argmax(A, axis=1)
    np.testing.assert_array_equal(max_idx, expected_max)

    # Test 2: Without return_indices
    min_val = linalg.min(A_tensor, dim=1, keepdims=False, return_indices=False)
    expected_min_val = np.min(A, axis=1)
    np.testing.assert_array_equal(min_val, expected_min_val)

    max_val = linalg.max(A_tensor, dim=1, keepdims=False, return_indices=False)
    expected_max_val = np.max(A, axis=1)
    np.testing.assert_array_equal(max_val, expected_max_val)

    # Test 3: With keepdims=True
    min_val, min_idx = linalg.min(A_tensor, dim=1, keepdims=True, return_indices=True)
    expected_min = np.argmin(A, axis=1, keepdims=True)
    np.testing.assert_array_equal(min_idx, expected_min)
    assert min_val.shape == (2, 1, 3)

    max_val, max_idx = linalg.max(A_tensor, dim=1, keepdims=True, return_indices=True)
    expected_max = np.argmax(A, axis=1, keepdims=True)
    np.testing.assert_array_equal(max_idx, expected_max)
    assert max_val.shape == (2, 1, 3)

    # Test 4: Without dim (global min/max)
    min_val = linalg.min(A_tensor, dim=None, keepdims=False, return_indices=False)
    expected_min_val = np.min(A)
    np.testing.assert_array_equal(min_val, expected_min_val)

    max_val = linalg.max(A_tensor, dim=None, keepdims=False, return_indices=False)
    expected_max_val = np.max(A)
    np.testing.assert_array_equal(max_val, expected_max_val)

    # Test 5: Global min/max with return_indices should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        linalg.min(A_tensor, dim=None, keepdims=False, return_indices=True)
    with pytest.raises(NotImplementedError):
        linalg.max(A_tensor, dim=None, keepdims=False, return_indices=True)

    # Test 6: Global min/max with keepdims=True
    min_val = linalg.min(A_tensor, dim=None, keepdims=True, return_indices=False)
    expected_min_val = np.min(A, keepdims=True)
    np.testing.assert_array_equal(min_val, expected_min_val)

    max_val = linalg.max(A_tensor, dim=None, keepdims=True, return_indices=False)
    expected_max_val = np.max(A, keepdims=True)
    np.testing.assert_array_equal(max_val, expected_max_val)

# -------------------------------------------------------------------
# Test logsumexp
# -------------------------------------------------------------------
def test_logsumexp(linalg):
    arr = np.array([1, 2, 3])
    arr_tensor = linalg.get_tensor(values=arr)
    result = linalg.logsumexp(arr_tensor, dim=0)
    expected = scipy.special.logsumexp(arr)
    if linalg.backend == "numpy":
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    else:
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

# -------------------------------------------------------------------
# Test basic tensor operations: permutax, transax, unsqueeze, cat.
# -------------------------------------------------------------------
def test_tensor_ops(linalg):
    # Test permutax and transax
    A = np.array([[1, 2],
                  [3, 4]])
    A_tensor = linalg.get_tensor(values=A)
    
    B = linalg.permutax(A_tensor, 1, 0)
    expected = A.T
    np.testing.assert_array_equal(linalg.tonp(B), expected)
    
    C = linalg.transax(A_tensor, 0, 1)
    np.testing.assert_array_equal(linalg.tonp(C), expected)
    
    # Test unsqueeze
    D = linalg.unsqueeze(A_tensor, 0)
    expected = np.expand_dims(A, 0)
    np.testing.assert_array_equal(linalg.tonp(D), expected)
    
    # Test cat
    E = linalg.cat([A_tensor, A_tensor], dim=0)
    expected = np.concatenate([A, A], axis=0)
    np.testing.assert_array_equal(linalg.tonp(E), expected)

# -------------------------------------------------------------------
# Test probability functions: norm_pdf, norm_log_pdf, norm_cdf, norm_inv_cdf.
# -------------------------------------------------------------------
def test_probability_functions(linalg):
    x = np.array([0.0, 0.5, 1.0])
    x_tensor = linalg.get_tensor(values=x)
    std_tensor = linalg.get_tensor(values=np.array(1.0))
    
    pdf = linalg.norm_pdf(x_tensor, mu=0.0, std=std_tensor)
    log_pdf = linalg.norm_log_pdf(x_tensor, mu=0.0, std=std_tensor)
    cdf = linalg.norm_cdf(x_tensor, mu=0.0, std=std_tensor)
    inv_cdf = linalg.norm_inv_cdf(cdf, mu=0.0, std=std_tensor)
    
    if linalg.backend == "numpy":
        np.testing.assert_allclose(inv_cdf, x, rtol=1e-5)
        np.testing.assert_allclose(log_pdf, np.log(pdf), rtol=1e-5)
    else:
        np.testing.assert_allclose(inv_cdf.cpu().numpy(), x, rtol=1e-5)
        np.testing.assert_allclose(log_pdf.cpu().numpy(), np.log(pdf.cpu().numpy()), rtol=1e-5)

# -------------------------------------------------------------------
# Test min_pos_max_neg
# Note: This function is designed to work with float arrays only.
# -------------------------------------------------------------------
def test_min_pos_max_neg(linalg):
    # Test 1: Basic case with positive and negative values
    A = np.array([[-2.0, 1.0, 3.0],
                  [-1.0, 2.0, -3.0]], dtype=np.float64)
    A_tensor = linalg.get_tensor(values=A)
    
    # Test return_what=1 (max positive)
    result = linalg.min_pos_max_neg(A_tensor, return_what=1)
    expected = np.array([1.0, 2.0])  # min of positive values in each row
    np.testing.assert_array_equal(result, expected)
    
    # Test return_what=-1 (min negative)
    result = linalg.min_pos_max_neg(A_tensor, return_what=-1)
    expected = np.array([-2.0, -1.0])  # max of negative values in each row
    np.testing.assert_array_equal(result, expected)
    
    # Test return_what=0 (both)
    min_neg, max_pos = linalg.min_pos_max_neg(A_tensor, return_what=0)
    expected_min_neg = np.array([-2.0, -1.0])
    expected_max_pos = np.array([1.0, 2.0])
    np.testing.assert_array_equal(min_neg, expected_min_neg)
    np.testing.assert_array_equal(max_pos, expected_max_pos)
    
    # Test 2: With keepdims=True
    result = linalg.min_pos_max_neg(A_tensor, return_what=1, keepdims=True)
    expected = np.array([[1.0], [2.0]])  # min of positive values in each row
    np.testing.assert_array_equal(result, expected)
    
    # Test 3: With return_indices=True
    result, indices = linalg.min_pos_max_neg(A_tensor, return_what=1, return_indices=True)
    expected = np.array([1.0, 2.0])
    expected_indices = np.array([1, 1])  # indices of min positive values
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(indices, expected_indices)
    
    # Test 4: All positive values
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float64)
    A_tensor = linalg.get_tensor(values=A)
    result = linalg.min_pos_max_neg(A_tensor, return_what=-1)
    expected = np.array([-np.inf, -np.inf])  # no negative values
    np.testing.assert_array_equal(result, expected)
    
    # Test 5: All negative values
    A = np.array([[-1.0, -2.0, -3.0],
                  [-4.0, -5.0, -6.0]], dtype=np.float64)
    A_tensor = linalg.get_tensor(values=A)
    result = linalg.min_pos_max_neg(A_tensor, return_what=1)
    expected = np.array([np.inf, np.inf])  # no positive values
    np.testing.assert_array_equal(result, expected)

# -------------------------------------------------------------------
# Test tensormul_T
# -------------------------------------------------------------------
def test_tensormul_T(linalg):
    # Test 1: Basic 2D matrix multiplication
    A = np.array([[1, 2],
                  [3, 4]])
    x = np.array([[5, 6],
                  [7, 8]])
    A_tensor = linalg.get_tensor(values=A)
    x_tensor = linalg.get_tensor(values=x)

    # tensormul_T does: (A @ x.T).T
    result = linalg.tensormul_T(A_tensor, x_tensor)
    expected = (A @ x.T).T
    np.testing.assert_array_equal(result, expected)

    # Test 2: 3D tensors with default dimensions (-2, -1)
    A = np.array([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
    x = np.array([[[9, 10],
                   [11, 12]],
                  [[13, 14],
                   [15, 16]]])
    A_tensor = linalg.get_tensor(values=A)
    x_tensor = linalg.get_tensor(values=x)

    # For each batch:
    # tensormul_T does: (A[i] @ x[i].T).T
    result = linalg.tensormul_T(A_tensor, x_tensor)
    expected = np.array([(A[0] @ x[0].T).T,
                        (A[1] @ x[1].T).T])
    np.testing.assert_array_equal(result, expected)

    # Test 3: 3D tensors with custom dimensions (0, 1)
    A = np.array([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
    x = np.array([[[9, 10],
                   [11, 12]],
                  [[13, 14],
                   [15, 16]]])
    A_tensor = linalg.get_tensor(values=A)
    x_tensor = linalg.get_tensor(values=x)

    # When dim0=0, dim1=1, we first swap axes 0 and 1 of x
    # Then multiply with A, then swap axes of the result back
    result = linalg.tensormul_T(A_tensor, x_tensor, dim0=0, dim1=1)

    # Calculate expected result:
    # 1. First swap axes 0 and 1 of x
    x_swapped = np.swapaxes(x, 0, 1)
    # 2. Multiply A with x_swapped (batch matrix multiplication)
    mul_result = A @ x_swapped
    # 3. Swap axes of the result back
    expected = np.swapaxes(mul_result, 0, 1)

    np.testing.assert_array_equal(result, expected)