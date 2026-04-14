import numpy as np
import tables as pt
import sys
import gc
import re
# np.seterr(all='raise')
from cobra.io import save_json_model
from PolyRound.api import Polytope
from sbmfi.core.metabolite import LabelledMetabolite
import pandas as pd
import functools
import traceback
# import line_profiler
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from sbmfi.config import SBMFIConfig
# poetry add git+https://github.com/GeomScale/dingo.git
# poetry install git+https://github.com/GeomScale/dingo.git
# TODO mail this dude for postdoc in Barca: https://torres-sanchez.xyz/
# buncha regexes
_bigg_compartment_ids = [
    'c', 'e', 'p', 'm', 'x', 'r', 'v', 'n',
    'g', 'u', 'l', 'h', 'f', 's', 'im',
    'cx', 'um', 'cm', 'i', 'mm', 'w', 'y',
]
_strip_bigg_rex         = re.compile(f'(_[{"|".join(_bigg_compartment_ids)}])$')
_find_biomass_rex       = re.compile('(biomass)', flags=re.IGNORECASE)
_read_atom_map_str_rex  = re.compile(r'(.*?)([-=<>]{3})(.*$)')
_net_constraint_rex     = re.compile('(?:_net)(?:|\|lb|\|ub)$')
_optlang_reverse_id_rex = re.compile('(_reverse_.{5})$')
_rev_reactions_rex      = re.compile('_rev$')
_xch_reactions_rex      = re.compile('_xch$')
_rho_constraints_rex    = re.compile('(?:_rho)(?:|_min|_max)(?:|\|lb|\|ub)$')
_biomass_coeff_rex      = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*([A-Za-z_]\w*)')
_get_dictlist_idxs = np.vectorize(lambda dctlst, item: dctlst.index(id=item.id), excluded=[0], otypes=[np.int64])


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------

def with_generator(fn):
    """
    Decorator that injects a fresh ``torch.Generator`` when ``generator=None``
    is passed (or omitted).  Apply to any function that calls torch random ops.

    Example
    -------
    >>> @with_generator
    ... def my_sample(n, generator):
    ...     return torch.randn(n, generator=generator)
    >>> my_sample(3)                              # reproducible on its own
    >>> my_sample(3, generator=g)                 # caller controls the seed
    """
    @functools.wraps(fn)
    def wrapper(*args, generator: Optional[torch.Generator] = None, **kwargs):
        if generator is None:
            generator = torch.Generator()
        return fn(*args, generator=generator, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Sparse → dense tensor construction
# ---------------------------------------------------------------------------

def _merge_duplicate_indices(
    indices: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge duplicate index rows by summing their corresponding values.

    Parameters
    ----------
    indices : np.ndarray, shape (N, D)
    values  : np.ndarray, shape (N,)

    Returns
    -------
    unique_indices : np.ndarray, shape (M, D)
    summed_values  : np.ndarray, shape (M,)
    """
    if values.size == 0:
        return indices, values
    uniq_indices, first_idx, counts = np.unique(
        indices, axis=0, return_index=True, return_counts=True
    )
    new_values = values[first_idx].copy()
    dup_mask = counts > 1
    if np.any(dup_mask):
        for i in np.where(dup_mask)[0]:
            mask = (indices == uniq_indices[i]).all(axis=1)
            new_values[i] = values[mask].sum()
    return uniq_indices, new_values


def get_tensor(
    shape: Optional[Tuple[int, ...]] = None,
    indices: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    dtype=None,
    config: Optional[SBMFIConfig] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Build a torch tensor, optionally scatter-filling from sparse (indices, values).

    Parameters
    ----------
    shape :
        Output tensor shape.  If *None* and no ``indices`` are given, the
        tensor is built directly from ``values``.
    indices :
        Integer array of shape (N, D) giving the coordinates of non-zero
        entries.  Duplicate rows are summed.
    values :
        Array of length N with the values to scatter.  Required when
        ``indices`` is provided.
    dtype :
        numpy or torch dtype.  Inferred from ``values`` when omitted.
    config :
        SBMFI runtime config carrying default device and dtype.

    Returns
    -------
    torch.Tensor
    """
    device = kwargs.pop('device', None)
    if kwargs:
        raise TypeError(f'unexpected keyword arguments: {list(kwargs.keys())}')

    if config is None:
        config = SBMFIConfig.from_env()
    if device is None:
        device = config.device

    # Resolve dtype → torch.dtype
    _NP_TORCH = {
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

    if dtype is None:
        if values is not None and np.asarray(values).size:
            dtype = np.asarray(values).dtype.type
        elif config is not None:
            dtype = config.dtype
        else:
            dtype = np.double
    if not isinstance(dtype, torch.dtype):
        dtype = _NP_TORCH.get(dtype, torch.double)

    if shape is not None:
        A = torch.zeros(shape, dtype=dtype, device=device)
        if indices is None:
            if values is not None:
                A[:] = torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
        elif np.asarray(indices).size:
            indices, values = _merge_duplicate_indices(
                np.asarray(indices), np.asarray(values)
            )
            idx = torch.as_tensor(indices, dtype=torch.int64, device=device)
            A[tuple(idx.T)] = torch.as_tensor(
                np.asarray(values), dtype=dtype, device=device
            )
    else:
        A = torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
    return A


# ---------------------------------------------------------------------------
# 1-D convolution (replaces LinAlg.convolve)
# ---------------------------------------------------------------------------

def convolve(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Full 1-D convolution matching ``np.convolve`` semantics.

    Parameters
    ----------
    a, v : torch.Tensor
        1-D tensors, **or** 2-D tensors of shape (batch, length) in which case
        each row is convolved independently.

    Returns
    -------
    torch.Tensor
        If inputs are 1-D: shape ``(len(a) + len(v) - 1,)``.
        If inputs are 2-D: shape ``(batch, len_a + len_v - 1)``.
    """
    if a.ndim == 1:
        # (1, 1, L) × (1, 1, K) → (1, 1, L+K-1) → (L+K-1,)
        a_ = a.view(1, 1, -1)
        v_ = v.view(1, 1, -1).flip(2)
        return F.conv1d(a_, v_, padding=v_.size(2) - 1).squeeze()
    elif a.ndim == 2:
        # (1, B, L) × (B, 1, K) → groups=B → (1, B, L+K-1) → (B, L+K-1)
        a_ = a.unsqueeze(0)              # (1, B, L)
        v_ = v.unsqueeze(1).flip(2)      # (B, 1, K)
        return F.conv1d(a_, v_, padding=v_.size(2) - 1, groups=a.size(0)).squeeze(0)
    else:
        raise ValueError(f"convolve expects 1-D or 2-D tensors, got shape {a.shape}")


# ---------------------------------------------------------------------------
# min_pos / max_neg helper (replaces LinAlg.min_pos_max_neg)
# ---------------------------------------------------------------------------

def min_pos_max_neg(
    alpha: torch.Tensor,
    return_what: int = 0,
    keepdims: bool = False,
    return_indices: bool = False,
):
    """
    Return the minimum positive and / or maximum negative values along the
    last dimension of *alpha*.

    Parameters
    ----------
    return_what :
        ``1``  → only α_max (min of positives),
        ``-1`` → only α_min (max of negatives),
        ``0``  → both as ``(α_min, α_max)``.
    """
    inf = float("inf")
    if return_what >= 0:
        pos = alpha.clone()
        pos[pos <= 0.0] = inf
        if return_indices:
            alpha_max, idx_max = pos.min(dim=-1, keepdim=keepdims)
        else:
            alpha_max = pos.min(dim=-1, keepdim=keepdims).values
        if return_what == 1:
            return (alpha_max, idx_max) if return_indices else alpha_max
    if return_what <= 0:
        neg = alpha.clone()
        neg[neg >= 0.0] = -inf
        if return_indices:
            alpha_min, idx_min = neg.max(dim=-1, keepdim=keepdims)
        else:
            alpha_min = neg.max(dim=-1, keepdim=keepdims).values
        if return_what == -1:
            return (alpha_min, idx_min) if return_indices else alpha_min
    if return_indices:
        return (alpha_min, idx_min), (alpha_max, idx_max)
    return alpha_min, alpha_max


# ---------------------------------------------------------------------------
# tensormul_T  (replaces LinAlg.tensormul_T)
# ---------------------------------------------------------------------------

def tensormul_T(
    A: torch.Tensor,
    x: torch.Tensor,
    dim0: int = -2,
    dim1: int = -1,
) -> torch.Tensor:
    """
    Compute ``(A @ x.T).T`` with transposition along *dim0* / *dim1*.

    Equivalent to ``torch.transpose(A @ torch.transpose(x, dim0, dim1), dim0, dim1)``.
    Used heavily in coordinater.py and polytopia.py.
    """
    return torch.transpose(A @ torch.transpose(x, dim0, dim1), dim0, dim1)


# ---------------------------------------------------------------------------
# scale  (replaces LinAlg.scale)
# ---------------------------------------------------------------------------

def scale(
    A: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    rev: bool = False,
) -> torch.Tensor:
    """
    Linearly scale A between *lo* and *hi*.

    ``rev=False`` (forward):  ``(A - lo) / (hi - lo)``  → maps [lo, hi] → [0, 1]
    ``rev=True``  (inverse):  ``A * (hi - lo) + lo``     → maps [0, 1] → [lo, hi]
    """
    return A * (hi - lo) + lo if rev else (A - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# sample_unit_hyper_sphere_ball  (replaces LinAlg.sample_unit_hyper_sphere_ball)
# ---------------------------------------------------------------------------

@with_generator
def sample_unit_hyper_sphere_ball(
    shape: Tuple[int, ...],
    hemi: bool = False,
    ball: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample uniformly from the surface of a unit hypersphere, or (if ``ball=True``)
    uniformly from its interior.

    Parameters
    ----------
    shape :
        ``(*batch_dims, n_dim)`` — the last dimension is the space dimension.
    hemi :
        If True, reflect the first coordinate to the positive half-sphere.
    ball :
        If True, return interior samples (uniform in the ball).
    generator :
        Optional torch.Generator for reproducibility.
    """
    rnd = torch.randn(shape, generator=generator, dtype=torch.double)
    norm = torch.norm(rnd, p=2, dim=-1, keepdim=True)
    sphere = rnd / norm
    if hemi:
        sphere[..., 0] = sphere[..., 0].abs()
    if ball:
        u = torch.rand((*shape[:-1], 1), generator=generator, dtype=torch.double)
        r = u ** (1.0 / shape[-1])
        return r * sphere
    return sphere


def tonp(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a NumPy array (CPU, no-grad assumed)."""
    return x.numpy()


def ref_counter(objct):
    print(sys.getrefcount(objct))
    referrers = gc.get_referrers(objct)
    print("Objects referring to a:")
    for ref in referrers:
        print(id(ref), ref)


def make_multidex(index_dict: Dict[str, pd.Index], name0='labelling_id', name1=None):
    if name1 is None:
        index1 = list(index_dict.values())[0]
        if index1 is not None:
            name1 = index1.name
    return pd.MultiIndex.from_frame(
        pd.DataFrame.from_dict(index_dict, orient='index').stack().reset_index().iloc[:, [0, 2]]
        , names=[name0, name1])


# def profile(profiler: line_profiler.LineProfiler):
#     def outer(func):  # profiler.print_stats()
#         def inner(*args, **kwargs):
#             profiler.add_function(func)
#             profiler.enable_by_count()
#             return func(*args, **kwargs)
#         return inner
#     return outer


def stacktrace(func):
    # https://stackoverflow.com/questions/1156023/print-current-call-stack-from-a-method-in-code
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        # Get all but last line returned by traceback.format_stack()
        # which is the line below.
        callstack = '\n'.join([4*' '+ line.strip() for line in traceback.format_stack()][:-1])
        print('{}() called:'.format(func.__name__))
        print(callstack)
        return func(*args, **kwds)

    return wrapped

# very general functions to get the most important dataframes for a HDF file with spectra stored
def hdf_opener_and_closer(mode='r'):
    def outer(func):
        def inner(*args, **kwargs):
            # pt.file._open_files.close_all()
            close = False
            hdf = kwargs.get('hdf', None)
            if hdf is None:
                argi = 0
                if not isinstance(args[0], str):  # checks whether a function is a bound to a class or not
                    argi = 1
                hdf = args[argi]
            if isinstance(hdf, str):
                close = True
                hdf = pt.open_file(hdf, mode=mode)
            in_kwargs = kwargs.get('hdf', None)
            if in_kwargs is None:
                args = list(args)
                args[argi] = hdf
            else:
                kwargs['hdf'] = hdf
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                hdf.close()
                raise e
            if close:
                hdf.close()
            return result
        return inner
    return outer


def _cell_color(x):
    # can also combine args into f'background-color: {hex1}; color: {hex2}'
    if x < 0:
        color = '#a6f1a6'
    elif x > 0:
        color = 'yellow'
    else:
        return 'color: white'
    return 'background-color : ' + color


def excel_polytope(pol: Polytope, file, concat=True):
    # debugging functions to manually check system
    # TODO also store the affine transformations! Only useful if we keep the index names, which PolyRound does not currently do
    ew = pd.ExcelWriter(path=file, engine='xlsxwriter')
    pol.b.name = 'b'
    if concat:
        blank = pd.Series(0, index=pol.b.index, name='blank')
        Ab = pd.concat([pol.A, blank, pol.b], axis=1)

        # blank2 = pd.Series(0, index=pol.b.index, name='blank')
        # Ttau = pd.concat([pol.transformation.T, pol.transformation.T])
        try:
            Abs = Ab.style.applymap(lambda x: _cell_color(x))
            Abs.to_excel(ew, sheet_name='Ab')
        except:
            Ab.to_excel(ew, sheet_name='Ab')
    else:
        A = pol.A.style.applymap(lambda x: _cell_color(x))
        A.to_excel(ew, sheet_name='A')
        pol.b.to_excel(ew, sheet_name='b')
    if pol.S is not None:
        pol.h.name = 'h'
        if concat:
            blank = pd.Series(0, index=pol.h.index, name='blank')
            Sh = pd.concat([pol.S, blank, pol.h], axis=1).style.applymap(lambda x: _cell_color(x))
            Sh.to_excel(ew, sheet_name='Sh')
        else:
            S = pol.S.style.applymap(lambda x: _cell_color(x))
            S.to_excel(ew, sheet_name='S')
            pol.h.to_excel(ew, sheet_name='h')
    ew.save()


def e_coli_core_escher_map(model, fluxes, filename='map.html'):
    from escher import Builder
    json_file = os.path.join(MODEL_DIR, 'escher_input', 'model', '___escher_placholder___.json')
    assert len(fluxes)%2 == 0, 'number of fluxes needs to be even, meaning fwd & rev split!'
    assert fluxes.index.str.contains('_rev$', regex=True).sum() == len(fluxes)/2, \
        'every reac needs a reverse variable'

    fwd_fluxes = fluxes.iloc[range(0, len(fluxes), 2)]
    rev_fluxes = fluxes.iloc[range(1, len(fluxes), 2)]
    net_fluxes = fwd_fluxes - rev_fluxes.values
    net_fluxes.name = fluxes.name
    save_json_model(model=model.make_sbml_writable(), filename=json_file, sort=False)
    builder = Builder(
        model_json=json_file,
        map_json=os.path.join(MODEL_DIR, 'escher_input', 'map', 'e_coli_core_tomek.json'),
        reaction_data=net_fluxes,
    )
    os.remove(json_file)
    builder.save_html(filename)


# method to get naturally labelled abundance vector for arbitrary susbtrate
def generate_nat_labelled_isotop_state(metabolite: 'LabelledMetabolite', min_abundance=1e-3):
    raise NotImplementedError


if __name__ == "__main__":
    import os
