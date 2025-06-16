"""
Unified backend API for numerical linear algebra operations using NumPy or PyTorch.

This module provides a common interface for many linear algebra and probability functions,
supporting both NumPy and PyTorch as backends. It includes utilities for LU decompositions,
tensor creation, convolution, and probability density functions among others.

Usage:
    nl = LinAlg(backend='numpy', seed=42)
    A = nl.get_tensor(shape=(3, 3), indices=np.array([[0, 1], [1, 2]]), values=np.array([5, 10]), squeeze=False, dtype=np.float64, device=None)
    LU = nl.LU(A)
    x = nl.solve(LU, b)
"""

from __future__ import annotations
import copy
import math
import random
import inspect
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy
from scipy.special import expit, logit, logsumexp, erf, erfinv

# Global constants used for probability density functions.
_SQRT2PI = math.sqrt(2 * math.pi)
_ONEBYSQRT2PI = 1.0 / _SQRT2PI
_SQRT2 = math.sqrt(2)
_2PI = 2 * math.pi
_1_SQRT2 = 1.0 / _SQRT2
_LN2PI_2 = math.log(_2PI) / 2.0


def _conditional_torch_import() -> int:
    """
    Imports torch and sets up a mapping between NumPy and torch dtypes.

    Returns:
        int: The minor version number of torch (as an integer).

    Raises:
        ImportError: If torch is not installed.
    """
    try:
        global torch
        import torch  # type: ignore
        version = int(torch.__version__.split('.')[1])
    except ImportError as e:
        print("torch not installed, cannot use this backend")
        raise e

    global _NP_TORCH_DTYPE
    _NP_TORCH_DTYPE = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.double: torch.double,
        np.single: torch.float32,
    }
    return version


def _merge_duplicate_indices(
    indices: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge duplicate indices by summing the corresponding values.

    Args:
        indices (np.ndarray): Array of indices with shape (n, d).
        values (np.ndarray): Array of values with length n.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (unique_indices, summed_values)
            where unique_indices contains unique rows and summed_values contains the sum over duplicates.
    """
    if values.size == 0:
        return indices, values

    # Get unique indices, first occurrences, and counts.
    uniq_indices, first_idx, counts = np.unique(
        indices, axis=0, return_index=True, return_counts=True
    )
    new_values = values[first_idx].copy()
    # For indices appearing more than once, sum the corresponding values.
    dup_mask = counts > 1
    if np.any(dup_mask):
        for i in np.where(dup_mask)[0]:
            mask = (indices == uniq_indices[i]).all(axis=1)
            new_values[i] = values[mask].sum()
    return uniq_indices, new_values


class Backend:
    def __init__(self, seed: Optional[int] = None, dtype: type = np.double, fkwargs=None):
        kwargs = copy.deepcopy(self._DEFAULT_FKWARGS)
        if fkwargs is not None:
            for fname, instance_kwargs in fkwargs.items():
                if fname in kwargs:
                    kwargs[fname].update(instance_kwargs)
        self._fkwargs = kwargs
        self._def_dtype = dtype

    def _determine_dtype(self, values=None, dtype=None):
        """
        Determine the appropriate dtype for tensor creation.
        
        Args:
            values: Input values array (optional)
            dtype: Explicitly requested dtype (optional)
            
        Returns:
            The determined dtype
        """
        if dtype is not None:
            return dtype

        if values is not None and values.size:
            dtype = values.dtype.type
            return dtype

        return self._def_dtype


class NumpyBackend(Backend):
    """
    Backend implementation using NumPy (and SciPy) for linear algebra operations.
    """

    _DEFAULT_FKWARGS = {
        'LU': {'overwrite_a': True, 'check_finite': False},
        'solve': {'trans': 0, 'overwrite_b': True, 'check_finite': False},
    }
    _BATCH_PROCESSING = True

    def __init__(self, seed: Optional[int] = None, dtype: type = np.double, fkwargs=None, **kwargs: Any):
        """
        Initialize the NumPy backend.

        Args:
            seed (Optional[int]): Seed for the random number generator.
            dtype (type): Default data type for created arrays.
        """
        super(NumpyBackend, self).__init__(seed=seed, dtype=dtype, fkwargs=fkwargs)
        self._rng = np.random.default_rng(seed=seed)
        self._def_dtype = dtype

    def get_tensor(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        indices: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        dtype: Optional[Any] = None,
        device: Any = None,  # device is unused for NumPy
    ) -> np.ndarray:
        """
        Create a tensor (NumPy array) and optionally populate entries using indices and values.

        Args:
            shape (Optional[Tuple[int, ...]]): Desired shape of the tensor. If None, shape is inferred from values.
            indices (Optional[np.ndarray]): Array of indices. If None, creates a dense tensor.
            values (Optional[np.ndarray]): Array of values. Required if indices is provided.
            dtype (Optional[Any]): Data type of the array. If None, inferred from values or uses default.
            device: Unused for NumPy.

        Returns:
            np.ndarray: The created tensor.
        """
        dtype = self._determine_dtype(values, dtype)
        
        if shape is not None:
            A = np.zeros(shape=shape, dtype=dtype)
            if indices is None:
                if values is not None:
                    A[:] = values.astype(dtype)
            elif indices.size:
                indices, values = _merge_duplicate_indices(indices, values)
                A[tuple(indices.T)] = values.astype(dtype)
        else:
            A = np.array(values, dtype=dtype)
        return A

    @staticmethod
    def vecopy(A: np.ndarray) -> np.ndarray:
        """Return a copy of array A."""
        return A.copy()

    @staticmethod
    def convolve(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Perform a one-dimensional convolution.

        Args:
            a (np.ndarray): First array.
            v (np.ndarray): Second array.

        Returns:
            np.ndarray: The convolution result.
        """
        if a.ndim == 2:
            # Process each row separately.
            return np.array([NumpyBackend.convolve(a[i, :], v[i, :]) for i in range(a.shape[0])])
        return np.convolve(a, v)

    @staticmethod
    def nonzero(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return indices and values of nonzero elements in A.

        Args:
            A (np.ndarray): Input array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (indices, values).
        """
        nonzero_idx = A.nonzero()
        return np.array(nonzero_idx, dtype=int).T, A[nonzero_idx]

    @staticmethod
    def tonp(A: np.ndarray) -> np.ndarray:
        """Return the input as a NumPy array (identity for NumPy)."""
        return A

    @staticmethod
    def set_to(A: np.ndarray, vals: Union[float, np.ndarray]) -> np.ndarray:
        """Set all elements of A to the given value(s)."""
        A[:] = vals if isinstance(vals, (int, float)) else vals[:]
        return A

    @staticmethod
    def permutax(A: np.ndarray, *args: int) -> np.ndarray:
        """Permute the axes of A."""
        return A.transpose(*args)

    @staticmethod
    def transax(A: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
        """Swap axes dim0 and dim1 of A."""
        return np.swapaxes(A, dim0, dim1)

    @staticmethod
    def unsqueeze(A: np.ndarray, dim: int) -> np.ndarray:
        """Add a new axis at the specified dimension."""
        return np.expand_dims(A, dim)

    @staticmethod
    def cat(As: list[np.ndarray], dim: int = 0) -> np.ndarray:
        """Concatenate a list of arrays along the specified dimension."""
        return np.concatenate(As, axis=dim)

    @staticmethod
    def max(
        A: np.ndarray, dim: Optional[int] = None, keepdims: bool = False, return_indices: bool = False
    ):
        """
        Return the maximum value(s) of A along an axis.

        Args:
            A (np.ndarray): Input array.
            dim (Optional[int]): Dimension along which to compute the max.
            keepdims (bool): Whether to keep the dimensions.
            return_indices (bool): Whether to return the indices of max values.

        Returns:
            Either max values or a tuple (max_values, max_indices).
        """
        if dim is not None:
            max_values = A.max(axis=dim, keepdims=keepdims)
            if return_indices:
                max_indices = A.argmax(axis=dim)
                return max_values, max_indices
            return max_values
        return A.max()

    @staticmethod
    def min(
        A: np.ndarray, dim: Optional[int] = None, keepdims: bool = False, return_indices: bool = False
    ):
        """
        Return the minimum value(s) of A along an axis.

        Args:
            A (np.ndarray): Input array.
            dim (Optional[int]): Dimension along which to compute the min.
            keepdims (bool): Whether to keep the dimensions.
            return_indices (bool): Whether to return the indices of min values.

        Returns:
            Either min values or a tuple (min_values, min_indices).
        """
        if dim is not None:
            min_values = A.min(axis=dim, keepdims=keepdims)
            if return_indices:
                min_indices = A.argmin(axis=dim)
                return min_values, min_indices
            return min_values
        return A.min()

    @staticmethod
    def view(A: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape A to the given shape."""
        return A.reshape(shape)

    @staticmethod
    def logsumexp(A: np.ndarray, dim: int = 0, keepdims: bool = False) -> np.ndarray:
        """Compute the log-sum-exp of A along the specified dimension."""
        return logsumexp(A, axis=dim, keepdims=keepdims)

    @staticmethod
    def atan2(x, y):
        """Return the elementwise arctan of x/y using the signs of the arguments."""
        return np.arctan2(x, y)

    @staticmethod
    def triu_indices(n: int, k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Return the indices for the upper-triangle of an (n x n) array, offset by k."""
        return np.triu_indices(n=n, k=k)

    @staticmethod
    def diag_embed(A):
        if len(A.shape) > 2:
            raise ValueError
        result = A[..., None] * np.eye(A.shape[-1])
        return result

    def LU(self, A: np.ndarray) -> Union[Tuple, list]:
        """
        Compute the LU factorization of a matrix or batch of matrices.

        Args:
            A (np.ndarray): The input array. If 3D, LU is computed for each slice.
            **kwargs: Additional keyword arguments for SciPy's LU factorization.

        Returns:
            The LU factorization of A.
        """
        if A.ndim == 3:
            return [NumpyBackend.LU(A[i, :, :]) for i in range(A.shape[0])]
        return scipy.linalg.lu_factor(A, self._fkwargs['LU'])

    def solve(self, LU: Any, b: np.ndarray) -> np.ndarray:
        """
        Solve a linear system given an LU factorization.

        Args:
            LU: The LU factorization (or batch thereof).
            b (np.ndarray): Right-hand side.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The solution array.
        """
        if b.ndim == 3:
            solution = np.empty_like(b)
            for i in range(b.shape[0]):
                solution[i, :, :] = NumpyBackend.solve(LU[i], b[i, :, :], **self._fkwargs['solve'])
            return solution
        return scipy.linalg.lu_solve(LU, b, **self._fkwargs['solve'])

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> np.ndarray:
        """Return a zeros array of the given shape and dtype."""
        if dtype is None:
            dtype = self._def_dtype
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> np.ndarray:
        """Return an ones array of the given shape and dtype."""
        if dtype is None:
            dtype = self._def_dtype
        return np.ones(shape, dtype=dtype)

    def randn(self, shape: Tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
        """Return samples from the standard normal distribution."""
        return self._rng.standard_normal(shape).astype(dtype)

    def randu(self, shape: Tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
        """Return samples from the uniform distribution over [0, 1)."""
        return self._rng.random(shape).astype(dtype)

    def randperm(self, n: int) -> np.ndarray:
        """Return a random permutation of integers from 0 to n-1."""
        return self._rng.permutation(n)

    def multinomial(self, n: int, p: np.ndarray) -> np.ndarray:
        """
        Draw samples from a multinomial distribution.

        Args:
            n (int): Number of samples.
            p (np.ndarray): Probabilities (should sum to 1).

        Returns:
            np.ndarray: The drawn sample indices.
        """
        counts = self._rng.multinomial(1, p, size=(n, *p.shape[:-1]))
        return np.where(counts)[1]

    def choice(self, n: int, tot: int, replace: bool = False) -> np.ndarray:
        """Choose n elements from tot elements with or without replacement."""
        return self._rng.choice(tot, n, replace=replace)


class TorchBackend(Backend):
    """
    Backend implementation using PyTorch for linear algebra operations.
    """

    _DEFAULT_FKWARGS = {
        'LU': {},
        'solve': {},
    }
    _BATCH_PROCESSING = True

    def __init__(
        self,
        seed: Optional[int] = None,
        device: str = 'cpu',
        dtype: type = np.double, 
        fkwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        """
        Initialize the Torch backend.

        Args:
            seed (Optional[int]): Seed for random number generation.
            device (str): Device identifier ('cpu' or 'cuda').
            dtype (type): Default numerical type.
        """
        super(TorchBackend, self).__init__(seed=seed, dtype=dtype, fkwargs=fkwargs)
        _conditional_torch_import()
        self._def_dtype = _NP_TORCH_DTYPE[dtype]
        self._device = torch.device('cpu')
        if torch.cuda.is_available() and 'cuda' in device:
            self._device = torch.device(device)

        self._rng = torch.Generator(self._device)
        if isinstance(seed, int):
            self._rng.manual_seed(seed)

        # Set default tensor type.
        # if self._def_dtype == torch.double:
        #     torch.set_default_tensor_type(torch.DoubleTensor)
        # elif self._def_dtype == torch.float32:
        #     torch.set_default_tensor_type(torch.FloatTensor)
        torch.autograd.set_detect_anomaly(True)

    def get_tensor(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        indices: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        """
        Create a torch tensor and optionally populate it via indices and values.

        Args:
            shape (Optional[Tuple[int, ...]]): Desired shape. If None, shape is inferred from values.
            indices (Optional[np.ndarray]): Array of indices. If None, creates a dense tensor.
            values (Optional[np.ndarray]): Array of values. Required if indices is provided.
            dtype (Optional[Any]): Data type. If None, inferred from values or uses default.
            device (Optional[Any]): Device to use; defaults to self._device.

        Returns:
            torch.Tensor: The created tensor.
        """
        if device is None:
            device = self._device
        dtype = self._determine_dtype(values, dtype)
        if not isinstance(dtype, torch.dtype):
            dtype = _NP_TORCH_DTYPE[dtype]
            
        if shape is not None:
            A = torch.zeros(shape, dtype=dtype, device=device)
            if indices is None:
                if values is not None:
                    A[:] = torch.as_tensor(values, device=device, dtype=dtype)
            elif indices.size:
                indices, values = _merge_duplicate_indices(indices, values)
                indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
                A[tuple(indices.T)] = torch.as_tensor(values, device=device, dtype=dtype)
        else:
            A = torch.as_tensor(values, device=device, dtype=dtype)
        return A

    def LU(self, A):
        """
        Compute the LU factorization of A.
        (Placeholder: this is replaced by one of the static methods above.)
        """
        return torch.linalg.lu_factor(A)

    def solve(self, LU, b):
        """
        Solve a linear system given LU and right-hand side b.
        """
        if b.ndim == 1:
            b = torch.atleast_2d(b).T
        return torch.lu_solve(b, *LU)

    @staticmethod
    def vecopy(A: torch.Tensor) -> torch.Tensor:
        """Return a copy of the tensor A."""
        return A.clone()

    @staticmethod
    def convolve(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform a one-dimensional convolution.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Convolution kernel.

        Returns:
            torch.Tensor: Convolved tensor.
        """
        # Ensure proper dimensions: if x is 1D, add batch and channel dims.
        if x.ndim == 1:
            x = x.view(1, 1, -1)
            y = y.view(1, 1, -1).flip(2)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(1).flip(2)
        else:
            raise ValueError("Only 1D or 2D tensors are supported for convolution.")
        return torch.conv1d(x, y, padding=y.size(2) - 1, groups=x.size(1)).squeeze()

    @staticmethod
    def nonzero(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return indices and values of nonzero elements in A.

        Args:
            A (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (indices, values)
        """
        nonzero_idx = torch.nonzero(A)
        return nonzero_idx, A[tuple(nonzero_idx.T)]

    @staticmethod
    def tonp(A: torch.Tensor) -> np.ndarray:
        """
        Convert a torch tensor to a NumPy array.
        """
        if torch.is_tensor(A):
            return A.cpu().detach().numpy()
        return A

    @staticmethod
    def view(A: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape tensor A to the given shape."""
        return A.view(shape)

    @staticmethod
    def permutax(A: torch.Tensor, *args: int) -> torch.Tensor:
        """Permute dimensions of A."""
        return A.permute(*args)

    @staticmethod
    def transax(A: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
        """Swap dimensions dim0 and dim1 in A."""
        return A.transpose(dim0, dim1)

    @staticmethod
    def unsqueeze(A: torch.Tensor, dim: int) -> torch.Tensor:
        """Add a dimension to A at index dim."""
        return A.unsqueeze(dim)

    @staticmethod
    def cat(As: list[torch.Tensor], dim: int) -> torch.Tensor:
        """Concatenate a list of tensors along dimension dim."""
        return torch.cat(As, dim)

    @staticmethod
    def max(
        A: torch.Tensor, dim: Optional[int] = None, keepdims: bool = False, return_indices: bool = False
    ):
        """
        Return the maximum value(s) along a dimension.

        Args:
            A (torch.Tensor): Input tensor.
            dim (Optional[int]): Dimension along which to compute max.
            keepdims (bool): Whether to retain reduced dimensions.
            return_indices (bool): Whether to return indices.

        Returns:
            Either the max values or a tuple (values, indices).
        """
        if dim is not None:
            result = A.max(dim=dim, keepdim=keepdims)
            return result if return_indices else result.values
        return A.max()

    @staticmethod
    def min(
        A: torch.Tensor, dim: Optional[int] = None, keepdims: bool = False, return_indices: bool = False
    ):
        """
        Return the minimum value(s) along a dimension.

        Args:
            A (torch.Tensor): Input tensor.
            dim (Optional[int]): Dimension along which to compute min.
            keepdims (bool): Whether to retain reduced dimensions.
            return_indices (bool): Whether to return indices.

        Returns:
            Either the min values or a tuple (values, indices).
        """
        if dim is not None:
            result = A.min(dim=dim, keepdim=keepdims)
            return result if return_indices else result.values
        return A.min()

    @staticmethod
    def logsumexp(A: torch.Tensor, dim: int = 0, keepdims: bool = False) -> torch.Tensor:
        """Compute the log-sum-exp of A along dimension dim."""
        return torch.logsumexp(A, dim=dim, keepdim=keepdims)

    @staticmethod
    def triu_indices(n: int, k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the indices for the upper-triangle of an (n x n) tensor."""
        indices = torch.triu_indices(row=n, col=n, offset=k)
        return indices[0], indices[1]

    @staticmethod
    def diag_embed(A):
        if len(A.shape) > 2:
            raise ValueError
        return torch.diag_embed(A)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> torch.Tensor:
        """Return a tensor filled with zeros."""
        if dtype is None:
            dtype = self._def_dtype
        elif dtype in _NP_TORCH_DTYPE:
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> torch.Tensor:
        """Return a tensor filled with ones."""
        if dtype is None:
            dtype = self._def_dtype
        return torch.ones(shape, dtype=dtype, device=self._device)

    def multinomial(self, n: int, p: torch.Tensor, replace: bool = True) -> torch.Tensor:
        """Draw samples from a multinomial distribution."""
        return torch.multinomial(input=p, num_samples=n, generator=self._rng, replacement=replace)

    def randn(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> torch.Tensor:
        """Return samples from the standard normal distribution."""
        if dtype is None:
            dtype = self._def_dtype
        elif not isinstance(dtype, torch.dtype):
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.randn(shape, generator=self._rng, dtype=dtype, device=self._device)

    def randu(self, shape: Tuple[int, ...], dtype: Optional[Any] = np.double) -> torch.Tensor:
        """Return samples from the uniform distribution over [0, 1)."""
        if dtype is None:
            dtype = self._def_dtype
        elif not isinstance(dtype, torch.dtype):
            dtype = _NP_TORCH_DTYPE[dtype]
        return torch.rand(shape, generator=self._rng, dtype=dtype, device=self._device)

    def randperm(self, n: int) -> torch.Tensor:
        """Return a random permutation of integers from 0 to n-1."""
        return torch.randperm(n, generator=self._rng)

    def choice(self, n: int, tot: int, replace: bool = False) -> torch.Tensor:
        """
        Randomly choose n indices from tot possibilities.
        """
        probs = torch.ones(tot, device=self._device) / tot
        return self.multinomial(n, probs, replace=replace)


class LinAlg:
    """
    A unified linear algebra API that abstracts the backend (NumPy or PyTorch).

    The LinAlg class exposes many common operations (matrix factorization, convolution,
    probability density functions, etc.) with the same interface regardless of the
    backend used.
    """

    _SAME_SIGNATURE = [
        # Functions whose signatures are identical across backends.
        "exp", "log10", "log", "atleast_2d", "diag", "trace", "allclose", "where", "arange",
        "divide", "prod", "diagonal", "tile", "sqrt", "isclose", "sum", "mean", "amax", "linspace",
        "cov", "split", "linalg.svd", "linalg.norm", "linalg.pinv", "linalg.cholesky", "linalg.det",
        "minimum", "maximum", "cumsum", "argmin", "argmax", "clip", "eye", "stack",
        "special.erf", "special.erfinv", "special.expit", "special.logit",
        "argsort", "unique", "cov", "split", "arctan2", "sin", "cos", "sign", "diff", "nansum",
        "isnan", "float_power", "einsum", 'special.gammaln'
    ]

    def __init__(
        self,
        backend: str,
        batch_size: int = 1,
        device: str = "cpu",
        fkwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        dtype: type = np.double,
    ):
        """
        Initialize the LinAlg object.

        Args:
            backend (str): Either 'numpy' or 'torch'.
            batch_size (int): Batch size for processing.
            device (str): Device for PyTorch ('cpu' or 'cuda').
            fkwargs (Optional[Dict[str, Any]]): Additional kwargs for factorization routines.
            seed (Optional[int]): Random seed.
            dtype (type): Default floating point type.
        """
        random.seed(seed)
        np.random.seed(seed)

        if dtype not in (np.double, np.float64, np.float32, np.single):
            raise ValueError("Not a supported default float type")

        if backend == "numpy":
            self._BACKEND = NumpyBackend(seed=seed, dtype=dtype, fkwargs=fkwargs)
        elif backend == "torch":
            self._BACKEND = TorchBackend(seed=seed, device=device, dtype=dtype, fkwargs=fkwargs)
        else:
            raise ValueError("Invalid backend provided.")

        # Dynamically attach functions with the same signature from the chosen backend.
        functions = self._fill_functions(backend)
        for name, method in inspect.getmembers(self._BACKEND, predicate=inspect.isroutine):
            if not (name.startswith('__') and name.endswith('__')):
                if name in functions:
                    raise ValueError
                functions[name] = method

        self.__dict__.update(functions)

        self._batch_size = int(batch_size) if self._BACKEND._BATCH_PROCESSING and batch_size > 1 else 1

        # Merge user and default kwargs.

        self._backwargs = {
            "backend": backend,
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "fkwargs": self._BACKEND._fkwargs,
            "batch_size": self._batch_size,
        }

    def _fill_functions(self, backend: str) -> dict:
        """
        Dynamically extract functions that have the same signature from the backend.

        Args:
            backend (str): 'numpy' or 'torch'.

        Returns:
            dict: Mapping of function names to functions.
        """
        functions = {}
        for fname in LinAlg._SAME_SIGNATURE:
            parts = fname.split(".")
            if len(parts) == 1:
                package = np if backend == "numpy" else torch
                func_name = parts[0]
            elif len(parts) == 2:
                mod, func_name = parts
                if backend == "torch":
                    package = getattr(torch, mod)
                else:
                    package = scipy.special if mod == "special" else getattr(np, mod)
            else:
                continue
            functions[func_name] = getattr(package, func_name)
        return functions

    def __getstate__(self):
        return self._backwargs

    def __setstate__(self, state):
        la = LinAlg(**state)
        self.__dict__.update(**la.__dict__)

    @property
    def backend(self) -> str:
        return self._backwargs['backend']

    def sample_unit_hyper_sphere_ball(self, shape: Tuple[int, ...], hemi: bool = False, ball=False) -> Any:
        """
        Sample uniformly from the surface of a hypersphere.
        If ball=True, return a uniform sample from the inside of a unit ball

        Args:
            shape (Tuple[int, ...]): Output shape.
            radius (float): Radius of the hypersphere.

        Returns:
            Tensor: Samples on the hypersphere.
        """
        rnd = self.randn(shape)
        norm = self.norm(rnd, 2, -1, True)
        sphere_sample = rnd / norm
        if hemi:
            sphere_sample[0] = abs(sphere_sample[0])
        if ball:
            randu = self.randu((*shape[:-1], 1)) ** (1 / shape[-1])
            return randu * sphere_sample
        return sphere_sample

    def _compute_xi(self, A: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Compute the standardized variable (A - mu)/std."""
        return (A - mu) / std

    def norm_pdf(self, A: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Evaluate the normal probability density function."""
        xi = self._compute_xi(A, mu, std)
        return _ONEBYSQRT2PI * (1.0 / std) * self.exp(-0.5 * xi**2)

    def norm_log_pdf(self, A: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Evaluate the log of the normal PDF."""
        xi = self._compute_xi(A, mu, std)
        return -_LN2PI_2 - self.log(std) - 0.5 * xi**2

    def norm_cdf(self, A: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Evaluate the cumulative distribution function of the normal distribution."""
        xi = self._compute_xi(A, mu, std)
        return 0.5 + 0.5 * self.erf(xi * _1_SQRT2)

    def norm_inv_cdf(self, u: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Compute the inverse CDF (quantile function) of the normal distribution."""
        return mu + std * _SQRT2 * self.erfinv(2 * u - 1.0)

    def trunc_norm_pdf(self, A: Any, lo: Any, hi: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Evaluate the PDF of a truncated normal distribution."""
        norm_pdf_val = self.norm_pdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta = self.norm_cdf(hi, mu, std)
        return norm_pdf_val / (beta - alpha)

    def trunc_norm_log_pdf(self, A: Any, lo: Any, hi: Any, mu: float = 0.0, std: float = 1.0, *args, **kwargs) -> Any:
        """Evaluate the log-PDF of a truncated normal distribution."""
        norm_log_pdf_val = self.norm_log_pdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta = self.norm_cdf(hi, mu, std)
        return norm_log_pdf_val - self.log(beta - alpha)

    def trunc_norm_cdf(self, A: Any, lo: Any, hi: Any, mu: float = 0.0, std: float = 1.0) -> Any:
        """Evaluate the CDF of a truncated normal distribution."""
        norm_cdf_val = self.norm_cdf(A, mu, std)
        alpha = self.norm_cdf(lo, mu, std)
        beta = self.norm_cdf(hi, mu, std)
        return (norm_cdf_val - alpha) / (beta - alpha)

    def trunc_norm_inv_cdf(self, u: Any, lo: Any, hi: Any, mu: float = 0.0, std: float = 1.0, return_log_prob: bool = False) -> Any:
        """
        Inverse CDF sampling from a truncated normal distribution.

        Args:
            u: Uniform samples.
            lo: Lower bound.
            hi: Upper bound.
            mu: Mean.
            std: Standard deviation.
            return_log_prob: Whether to return log probabilities.

        Returns:
            Samples (and optionally log probabilities).
        """
        alpha = self.norm_cdf(lo, mu, std)
        beta = self.norm_cdf(hi, mu, std)
        uu = alpha + u * (beta - alpha)
        samples = self.norm_inv_cdf(uu, mu, std)
        if return_log_prob:
            log_prob = self.norm_log_pdf(samples, mu, std) - self.log(beta - alpha)
            return samples, log_prob
        return samples

    def unif_inv_cdf(self, u: Any, lo: float = 0.0, hi: float = 1.0, return_log_prob: bool = False) -> Any:
        """
        Inverse CDF sampling from a uniform distribution.

        Args:
            u: Uniform samples.
            lo: Lower bound.
            hi: Upper bound.
            return_log_prob: Whether to return log probabilities.

        Returns:
            Samples (and optionally log probabilities).
        """
        diff = hi - lo
        samples = lo + u * diff
        if return_log_prob:
            log_probs = self.log(self.ones(u.shape) / diff)
            return samples, log_probs
        return samples

    def unif_log_pdf(self, A: Any, lo: float = 0.0, hi: float = 1.0, mu: float = 0.0, *args, **kwargs) -> Any:
        """
        Evaluate the log-PDF of a uniform distribution.

        Args:
            A: Points at which to evaluate.
            lo: Lower bound.
            hi: Upper bound.
            mu: Unused; kept for API consistency.

        Returns:
            Log-probabilities.
        """
        out_shape = A.shape if isinstance(mu, float) else tuple(np.maximum(A.shape, mu.shape))
        return self.log(self.ones(out_shape) / (hi - lo))

    def sample_bounded_distribution(self, shape: tuple, lo: Any, hi: Any, mu: float = 0.0, std: float = 0.1, which: str = "unif", return_log_prob: bool = False) -> Any:
        """
        Sample from a distribution bounded between lo and hi.

        Args:
            shape (tuple): Sample shape.
            lo: Lower bound.
            hi: Upper bound.
            mu: Mean (used for 'gauss').
            std: Standard deviation (used for 'gauss').
            which (str): 'unif' or 'gauss'.
            return_log_prob (bool): Whether to return log-probabilities.

        Returns:
            Samples (and optionally log probabilities).
        """
        if lo.shape != hi.shape:
            raise ValueError(f"lo.shape: {lo.shape} != hi.shape: {hi.shape}")
        u = self.randu(shape=(*shape, *lo.shape))
        if which == "unif":
            return self.unif_inv_cdf(u, lo, hi, return_log_prob)
        elif which == "gauss":
            return self.trunc_norm_inv_cdf(u, lo, hi, mu, std, return_log_prob)
        else:
            raise ValueError("Unsupported distribution type.")

    def bounded_distribution_log_prob(self, x: Any, lo: Any, hi: Any, mu: Any, std: float = 0.1, which: str = "unif", old_is_new: bool = False, unsqueeze: bool = True, k: int = 1) -> Any:
        """
        Compute the log-probability of x under a bounded distribution.

        Args:
            x: Samples.
            lo: Lower bound.
            hi: Upper bound.
            mu: Mean.
            std: Standard deviation.
            which: 'unif' or 'gauss'.
            old_is_new: Option for different handling.
            unsqueeze: Whether to unsqueeze dimensions.
            k: Parameter for triu_indices.

        Returns:
            Log-probabilities.
        """
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have the same shape")
        if x.ndim != mu.ndim:
            raise ValueError("Dimension mismatch between x and mu")
        if unsqueeze:
            mu = self.unsqueeze(mu, 1)
            x = self.unsqueeze(x, 0)
        pdf = self.unif_log_pdf if which == "unif" else self.trunc_norm_log_pdf
        if old_is_new:
            rows, cols = self.triu_indices(x.shape[1], k=k)
            diags = self.arange(x.shape[1])
            if unsqueeze:
                uptri_x = x[0, cols]
                uptri_mu = mu[rows, 0]
            else:
                uptri_x = x[rows, cols]
                uptri_mu = mu[cols, rows]
            uptri_probs = pdf(uptri_x, lo, hi, uptri_mu, std)
            out_shape = tuple(np.maximum(x.shape, mu.shape))
            log_probs = self.get_tensor(shape=out_shape)
            log_probs[rows, cols] = uptri_probs
            log_probs = log_probs + self.transax(log_probs, dim0=0, dim1=1)
            log_probs[diags, diags] /= 2
            return log_probs
        else:
            return pdf(x, lo, hi, mu, std)

    def tensormul_T(self, A: Any, x: Any, dim0: int = -2, dim1: int = -1) -> Any:
        """
        Multiply A and the transpose of x along specified dimensions.

        Args:
            A: Left tensor.
            x: Right tensor.
            dim0: First dimension to swap.
            dim1: Second dimension to swap.

        Returns:
            The result of the multiplication.
        """
        res = A @ self.transax(x, dim0=dim0, dim1=dim1)
        return self.transax(res, dim0=dim0, dim1=dim1)

    def eval_std_normal(self, x: Any) -> Any:
        """Evaluate a standard normal function (for testing)."""
        return x ** 2 - _SQRT2PI

    def min_pos_max_neg(self, alpha: Any, return_what: int = 1, keepdims: bool = False, return_indices: bool = False) -> Any:
        """
        Compute the minimum positive and maximum negative values of alpha.

        Args:
            alpha: Input tensor.
            return_what: 1 for max positive, -1 for min negative, 0 for both.
            keepdims: Whether to keep dimensions.
            return_indices: Whether to return indices.

        Returns:
            The computed values.
        """
        inf = float("inf")
        if return_what > -1:
            alpha_max = self.vecopy(alpha)
            alpha_max[alpha_max <= 0.0] = inf
            alpha_max = self.min(alpha_max, -1, keepdims, return_indices)
            if return_what == 1:
                return alpha_max
        if return_what < 1:
            alpha_min = self.vecopy(alpha)
            alpha_min[alpha_min >= 0.0] = -inf
            alpha_min = self.max(alpha_min, -1, keepdims, return_indices)
            if return_what == -1:
                return alpha_min
        return alpha_min, alpha_max

    def scale(self, A: Any, lo: float, hi: float, rev: bool = False) -> Any:
        """
        Scale A between lo and hi.

        Args:
            A: Input tensor.
            lo: Lower bound.
            hi: Upper bound.
            rev: If True, perform the inverse scaling.

        Returns:
            Scaled tensor.
        """
        return A * (hi - lo) + lo if rev else (A - lo) / (hi - lo)

    def unsqueeze_like(self, A: Any, like: Any) -> Any:
        """
        Unsqueeze A until it has the same number of dimensions as 'like'.

        Args:
            A: Input tensor.
            like: Reference tensor.

        Returns:
            The unsqueezed tensor.
        """
        n_unsqueezes = like.ndim - A.ndim
        return A[(None,) * n_unsqueezes + (...,)]


if __name__ == "__main__":
    import pickle
    import timeit
    import cProfile
    import torch

    linalg = LinAlg(backend='torch', seed=123)
    shape = (3, 3)
    # Use indices to set the diagonal entries.
    int_values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = linalg.get_tensor(shape=(2, 2), indices=None,
                               values=int_values, dtype=None, device=None)
