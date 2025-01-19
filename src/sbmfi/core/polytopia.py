import numpy as np
from typing import Union, List
import math
import scipy
import pandas as pd
import contextlib
import io
from cobra import Reaction
import functools
from sympy import nsimplify, Matrix
from sympy.core.numbers import One
import cvxpy as cp
import cdd
import cdd.gmp
from sbmfi.core.util import _optlang_reverse_id_rex, _rho_constraints_rex, _net_constraint_rex, \
    _rev_reactions_rex, _xch_reactions_rex
from sbmfi.core.linalg import LinAlg
from sbmfi.core.reaction import LabellingReaction
import copy
from PolyRound.api import PolyRoundApi, Polytope, PolyRoundSettings
from PolyRound.static_classes.lp_utils import ChebyshevFinder
from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import MaximumVolumeEllipsoidFinder
from fractions import Fraction
from importlib.metadata import version
# import pypoman

class LabellingPolytope(Polytope):
    tolerance = 1e-12
    def __init__(
            self,
            A: pd.DataFrame,
            b: pd.Series,
            S: pd.DataFrame = None,
            h: pd.Series = None,
            mapper: dict = None,
            objective: dict = None,
            non_labelling_reactions: pd.Index = None
    ):
        if b is not None:
            b.name = 'ineq'
        if h is not None:
            b.name = 'eq'
        Polytope.__init__(self, A=A, b=b, S=S, h=h)
        self._mapper: dict = mapper if mapper else {}
        self._objective: dict = objective if objective else {}
        self._nlr = non_labelling_reactions if non_labelling_reactions is not None else pd.Index([])

    @property
    def mapper(self):
        return self._mapper.copy()

    @property
    def non_labelling_reactions(self):
        return self._nlr.copy()

    @property
    def objective(self):
        return self._objective.copy()

    @objective.setter
    def objective(self, val: dict):
        for k in val.keys():
            if k not in self.A.columns:
                raise ValueError(f'{k} not in the polytope columns')
        self._objective = val

    def generate_cvxpy_LP(self, solve=False):
        # objective_reactions = cobra.util.solver.linear_reaction_coefficients(model)
        # polytope = polytope.copy()
        n = self.A.shape[1]

        objective = np.zeros(n)
        for rid, coef in self._objective.items():
            objective[self.A.columns.get_loc(rid)] = coef

        objective = cp.Parameter(shape=objective.shape, value=objective)

        n_ineq = self.A.shape[0]

        v_cp = cp.Variable(n, name='fluxes')
        A_cp = cp.Parameter(self.A.shape, name='A', value=self.A.values)
        b_cp = cp.Parameter(n_ineq, name='b', value=self.b.values)
        constraints = [A_cp @ v_cp <= b_cp]

        cvx_result = {}
        if self.S is not None:
            n_met = self.S.shape[0]
            S_cp = cp.Parameter(self.S.shape, name='S', value=self.S.values)
            h_cp = cp.Parameter(n_met, name='h', value=self.h.values)
            constraints.append(S_cp @ v_cp == h_cp)
            cvx_result['S'] = S_cp
            cvx_result['h'] = h_cp

        problem = cp.Problem(objective=cp.Maximize(objective @ v_cp), constraints=constraints)

        cvx_result.update({
            'v': v_cp,  # easiest way to change bounds in the cvxpy problem
            'A': A_cp,  # easiest way to change bounds in the cvxpy problem
            'b': b_cp,
            'constraints': constraints,
            'problem': problem,
            'objective': objective,
            'polytope': self,
        })

        if solve and len(self._objective) > 0:
            problem.solve(solver=cp.GUROBI, verbose=False)
            cvx_result['solution'] = pd.Series(v_cp.value, index=self.A.columns, name=f'optimum', dtype=np.float64)
            cvx_result['optimum'] = problem.value
        return cvx_result

    @staticmethod
    def from_Polytope(polytope:Polytope, labellingpolytope: 'LabellingPolytope' = None):
        kwargs = {}
        if labellingpolytope is not None:
            kwargs = {
                'mapper': labellingpolytope.mapper,
                'objective': labellingpolytope.objective,
                'non_labelling_reactions': labellingpolytope.non_labelling_reactions,
            }

        pol = LabellingPolytope(polytope.A, polytope.b, polytope.S, polytope.h, **kwargs)
        pol.transformation = polytope.transformation
        pol.shift = polytope.shift
        return pol


def fast_FVA(polytope: LabellingPolytope, full=False):
    cvx_result = polytope.generate_cvxpy_LP()
    problem = cvx_result['problem']
    objective = cvx_result['objective']
    polytope = cvx_result['polytope']
    objective.value[:] = 0.0

    result = {}
    for i, reaction_id in zip(range(objective.value.shape[0]), polytope.A.columns):
        objective.value[i] = 1.0
        problem.solve(solver=cp.GUROBI, ignore_dpp=True)
        if problem.status != 'optimal':
            # raise ValueError(f'{reaction_id}: {problem.status}')
            print(f'{reaction_id}: {problem.status}')
        reac_max = round(problem.value, 4)
        objective.value[i] = -1.0
        problem.solve(solver=cp.GUROBI, ignore_dpp=True)
        reac_min = round(problem.value * -1, 4)
        objective.value[i] = 0.0
        if full:
            # TODO store the full flux vector at optimum instead of only the optimum
            raise NotImplementedError
        else:
            result[reaction_id] = (reac_min, reac_max)
    return pd.DataFrame(result, index=['min', 'max']).T


def _set_cdd_lib(number_type='float'):
    global _CDD
    if number_type == 'float':
        _CDD = cdd
    elif number_type == 'fraction':
        _CDD = cdd.gmp


def _cdd_mat_pol(ineq, eq=None, return_polyhedron=True, number_type='float', reptype=cdd.RepType.INEQUALITY, canonicalize=True):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]

    (A, b) = ineq
    if b.ndim < 2:
        b = b[:, None]
    np_arr = np.hstack([b, -A])

    lin_set = ()
    if eq is not None:
        (S, h) = eq
        if h.ndim < 2:
            h = h[:, None]
        if cdd_version[0] > 2:
            eq_arr = np.hstack([h, -S])  # TODO S or -S not sure
            np_arr = np.vstack([np_arr, eq_arr])
            lin_set = np.arange(b.shape[0], np_arr.shape[0])
    if cdd_version[0] > 2:  # changes from version 3:  https://pycddlib.readthedocs.io/en/stable/changes.html#version-3-0-0-4-october-2024
        _set_cdd_lib(number_type)
        if number_type == 'fraction':
            fractionater = np.vectorize(lambda x: Fraction(x))
            np_arr = fractionater(np_arr)
        matrix = _CDD.matrix_from_array(np_arr, rep_type=reptype, lin_set=lin_set)
        if canonicalize:
            sets = _CDD.matrix_canonicalize(matrix)  # TODO for some reason lin_set is set here????
            matrix.lin_set = ()  # TODO THIS IS WEIRD
        return _CDD.polyhedron_from_matrix(matrix) if return_polyhedron else matrix
    else:
        matrix = cdd.Matrix(np_arr, number_type=number_type)
        matrix.rep_type = reptype
        matrix.lin_set = lin_set
        if canonicalize:
            matrix = matrix.canonicalize()
        return cdd.Polyhedron(matrix) if return_polyhedron else matrix


def _cdd_dual(polyhedron: cdd.Polyhedron | cdd.gmp.Polyhedron, to_V_rep=True):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]
    if cdd_version[0] > 2:
        return _CDD.copy_output(polyhedron)
    else:
        if to_V_rep:
            return polyhedron.get_generators()
        return polyhedron.get_inequalities()


def _cdd_mat_ar(mat: cdd.Matrix | cdd.gmp.Matrix):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]
    if cdd_version[0] > 2:
        mat = np.array(mat.array).astype(float)
    return np.array(mat)


# function copied from https://github.com/stephane-caron/pypoman
def project_polyhedron(proj, ineq, eq=None, canonicalize=True, number_type='float'):
    # raise ValueError('does not yet work after update of the pycddlib API for version >= 3.x.x')

    P = _cdd_mat_pol(
        ineq, eq, return_polyhedron=True, number_type=number_type,
        reptype=cdd.RepType.INEQUALITY, canonicalize=canonicalize
    )
    # OLD code
    # linsys = cdd.Matrix(np.hstack([b, -A]), number_type=number_type)
    # linsys.rep_type = cdd.RepType.INEQUALITY

    # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    # Convert from H- to V-representation
    generators = _cdd_dual(P)
    if generators.lin_set:
        print("Generators have linear set: {}".format(generators.lin_set))
    V = _cdd_mat_ar(generators)

    # Project output wrenches to 2D set
    (E, f) = proj
    vertices, rays = [], []
    free_coordinates = []
    for i in range(V.shape[0]):
        if generators.lin_set and i in generators.lin_set:
            free_coordinates.append(list(V[i, 1:]).index(1.))
        elif V[i, 0] == 1:  # vertex
            vertices.append(np.dot(E, V[i, 1:]) + f)
        else:  # ray
            rays.append(np.dot(E, V[i, 1:]))

    # print(13, vertices, rays)
    return vertices, rays


def compute_polytope_vertices(ineq, number_type='float'):
    P = _cdd_mat_pol(ineq, None, True, number_type, cdd.RepType.INEQUALITY)
    g = _cdd_dual(P)
    V = _cdd_mat_ar(g)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise Exception("Polyhedron is not a polytope")
        elif i not in g.lin_set:
            vertices.append(V[i, 1:])
    return vertices


def simplify_vertices(vertices: pd.DataFrame, tolerance=1e-10):
    vertices = vertices.drop_duplicates()
    diff = (vertices.values[None, ...] - vertices.values[:, None, ...])
    norm_tol = np.linalg.norm(diff, 2, 2) < tolerance
    n = vertices.shape[0]
    norm_tol[np.tril_indices(n)] = False
    rows, cols = np.where(norm_tol)
    selecta = np.ones(n, dtype=bool)
    selecta[cols] = False
    return vertices.loc[selecta]


def V_representation(polytope: Polytope, number_type='float', vertices_tol=1e-10):
    if polytope.S is not None:
        raise ValueError('must be in cannonical form!')
    vertices = compute_polytope_vertices(ineq=(polytope.A.values, polytope.b.values), number_type=number_type)
    # # TODO try bretl, somehow give it some equality constraints...
    # vertices = compute_polytope_vertices(A=polytope.A.values, b=polytope.b.values, number_type='float')
    # pypoman.duality.compute_polytope_vertices(A=self.A.values, b=self.b.values)
    vertices = pd.DataFrame(vertices, columns=polytope.A.columns)
    if number_type == 'float':
        vertices = vertices.astype(float)
    vertices[abs(vertices) < vertices_tol] = 0.0
    if vertices_tol > 0.0:
        vertices = simplify_vertices(vertices, vertices_tol)
    return vertices


def H_representation(vertices: List[np.array], number_type='float', halfspace_tol=1e-10):
    """
    Compute the halfspace representation (H-rep) of a polytope defined as
    convex hull of a set of vertices:

    .. math::

        A x \\leq b
        \\quad \\Leftrightarrow \\quad
        x \\in \\mathrm{conv}(\\mathrm{vertices})

    Parameters
    ----------
    vertices : list of arrays
        List of polytope vertices.

    Returns
    -------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
    """

    if isinstance(vertices, pd.DataFrame):
        V = vertices.values
    elif isinstance(vertices, list):
        V = np.vstack(vertices)
    t = np.ones((V.shape[0], 1))  # first column is 1 for vertices
    P = _cdd_mat_pol((-V, t), None,True, number_type, cdd.RepType.GENERATOR)
    bA = _cdd_dual(P, to_V_rep=False)
    bA = _cdd_mat_ar(bA)
    if bA.shape == (0,):  # bA == []
        return bA
    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])
    b[abs(b) < halfspace_tol] = 0.0
    A[abs(A) < halfspace_tol] = 0.0
    return A, b


def project_polytope(
        polytope,
        P: pd.DataFrame,
        p: pd.Series = None,
        return_vertices=False,
        vertices_tol=1e-10,
        number_type='float'
):
    # raise ValueError('not yet tested after pycddlib API change with version >= 3.x.x')
    # computes the projection/ infinite shadow of the labelling-polytope onto the exchange flux dimensions
    # https://github.com/stephane-caron/pypoman
    # https://pycddlib.readthedocs.io/en/latest/index.html
    P = P.loc[:, polytope.A.columns]
    if not P.index.isin(polytope.A.columns).all():
        raise ValueError('projection fluxes not in polytope fluxes')
    if p is None:
        p = pd.Series(0.0, index=P.index)
    ineq = (polytope.A.values, polytope.b.values)
    proj = (-P.values, p.values)
    if polytope.S is None:
        eq = None
    else:
        eq = (polytope.S.values, polytope.h.values)
    vertices, _ = project_polyhedron(proj=proj, ineq=ineq, eq=eq, number_type=number_type)
    vertices = pd.DataFrame(vertices, columns=P.index).drop_duplicates()  # TODO maybe clean up a bit
    if vertices_tol > 0.0:
        vertices = simplify_vertices(vertices, vertices_tol)
    if return_vertices:
        return vertices
    # A, b = pypoman.duality.compute_polytope_halfspaces(vertices.values)  # NOTE could also be done with scipy ConvHull
    A, b = H_representation(vertices)  # NOTE could also be done with scipy ConvHull
    A[abs(A) < vertices_tol] = 0.0
    # TODO make A have the right columns and come up with some index names
    return LabellingPolytope(A=pd.DataFrame(A, columns=P.index), b=pd.Series(b), mapper=None)


def rref_and_project(
        polytope: LabellingPolytope,
        P: pd.DataFrame,
        settings: PolyRoundSettings = PolyRoundSettings(),
        number_type='float',
        round=0,
):
    if not P.index.isin(polytope.A.columns).all():
        raise ValueError('projection fluxes not in polytope')
    polytope = polytope.copy()
    cols = polytope.A.columns
    if polytope.S is None:
        raise ValueError('need untransformed polytope')
    print(polytope.S)
    pppol, _, _, _ = transform_polytope_keep_transform(polytope, settings, kernel_id='rref')
    print(pppol.A)
    print(pppol.b)

    x, dist = ChebyshevFinder.chebyshev_center(polytope, settings)
    transformation, free_vars = rref_null_space(polytope.S, tolerance=settings.numerics_threshold)
    T_1 = pd.DataFrame(0.0, index=transformation.columns, columns=transformation.index)
    T_1.loc[transformation.columns, transformation.columns] = np.eye(len(free_vars))

    x_star = T_1 @ x
    tau = (x - (transformation @ x_star))
    tau[abs(tau) < settings.numerics_threshold] = 0.0
    polytope.apply_shift(tau.values)
    polytope.transformation.columns = cols
    polytope.apply_transformation(transformation)
    polytope.A.columns = transformation.columns
    if round > 0:
        polytope.A = polytope.A.round(round)
    print(12312312)
    print(polytope.A)
    print(polytope.b)
    return project_polytope(polytope, P, vertices_tol=settings.numerics_threshold, number_type=number_type)


def thermo_2_net_polytope(polytope: LabellingPolytope, verbose=False):
    if len(polytope.mapper) == 0: # NB this is already a net_pol
        if verbose:
            print('already net')
        return polytope

    if (_xch_reactions_rex.search(list(polytope.mapper.values())[0]) is None):
        raise ValueError('this is not a thermo_pol')

    A = polytope.A.loc[:, ~polytope.A.columns.isin(list(polytope.mapper.values()))]
    has_coeff = np.linalg.norm(A, axis=1) != 0.0
    A = A.loc[has_coeff]
    b = polytope.b.loc[has_coeff]

    S = None
    if polytope.S is not None:
        S = polytope.S.loc[:, A.columns]

    return LabellingPolytope(A=A, b=b, S=S, h=polytope.h, mapper=None, objective=polytope.objective)


def extract_labelling_polytope(
        model: 'LabellingModel',
        coordinates = 'labelling',
        zero_tol    = 1e-10,
        inf_bound   = 1e5,
) -> LabellingPolytope:
    # TODO test this thing with RatioMixins that have optlang ratio constraints! Should work, since we iterate over constraints
    if coordinates not in ['thermo', 'labelling']:
        raise ValueError('not a valid coordinate system')

    S_rows = {}
    h_rows = {}

    A_rows = {}
    b_rows = {}

    variables_id = dict((var.name, _optlang_reverse_id_rex.sub('_rev', var.name)) for var in model.variables)
    for constraint in model.constraints:
        # constraint.xb ∈ {value, inf, None}
        lb = constraint.lb
        ub = constraint.ub

        lb = lb if ((lb is None) or (abs(lb) != math.inf)) else inf_bound
        ub = ub if ((ub is None) or (abs(ub) != math.inf)) else inf_bound

        equality = False
        if (lb is not None) and (ub is not None) and ((ub - lb) < zero_tol):  # we know ub >= lb
            equality = True

        coefs = {
            variables_id[key.name]: val for key, val in
                constraint.get_linear_coefficients(constraint.variables).items()
        }

        if equality:
            S_rows[constraint.name] = coefs
            h_rows[constraint.name] = ub
        else:
            A_rows[constraint.name] = coefs
            b_rows[constraint.name] = [lb, ub]

    S = pd.DataFrame(S_rows, dtype=np.double).fillna(value=0.0).T  # metabolite and ratio equalities
    h = pd.Series(h_rows, name='eq')  # equalities

    A = pd.DataFrame(A_rows, index=S.columns, dtype=np.double).fillna(value=0.0).T  # these are the rho, net and ratio constraints
    b = pd.DataFrame(b_rows, index=['lb', 'ub']).T  # these are bounds on constraints in A

    bvar = {}
    non_labelling_reactions = []
    for reaction in model.reactions:
        if reaction.bounds == (0.0, 0.0):
            continue
        if type(reaction) == Reaction:
            non_labelling_reactions.append(reaction.id)
        fwd_var = reaction.forward_variable
        rev_var = reaction.reverse_variable
        if hasattr(reaction, '_rho_max') and (reaction.rho_max > 0.0):
            if coordinates == 'labelling':
                bvar[fwd_var.name] = (fwd_var.lb, fwd_var.ub)
                bvar[variables_id[rev_var.name]] = (rev_var.lb, rev_var.ub)
            elif coordinates == 'thermo':
                bvar[fwd_var.name] = reaction.bounds
                bvar[variables_id[rev_var.name]] = (reaction.rho_min, reaction.rho_max)
        # this causes a reaction that runs reverse to its 'definition' to be named reaction_rev in the polytope, which is not desirable
        elif (reaction.upper_bound <= 0.0) and (coordinates == 'labelling'):
            bvar[variables_id[rev_var.name]] = (-reaction.upper_bound, -reaction.lower_bound)
        else:
            bvar[reaction.id] = reaction.bounds

    non_labelling_reactions = pd.Index(non_labelling_reactions)

    bvar = pd.DataFrame(bvar, index=['lb', 'ub']).T
    A = A.loc[:, bvar.index]
    S = S.loc[:, bvar.index]

    if coordinates == 'thermo':
        wherho  = b.index.str.contains(_rho_constraints_rex)
        whernet = b.index.str.contains(_net_constraint_rex)
        A = A.loc[~(wherho | whernet)]
        b = b.loc[~(wherho | whernet)]

    n = bvar.shape[0]
    A_index = bvar.index

    wherrev = A_index.str.contains(_rev_reactions_rex)
    if coordinates == 'thermo':
        xchid = A_index[wherrev].str.replace(_rev_reactions_rex, '_xch', regex=True)
        mapper = dict([(k, v) for k, v in zip(A_index[wherrev], xchid)])
        A_index = A_index.map(lambda x: mapper[x] if x in mapper else x)
    else:
        fwdid = A_index[wherrev].str.replace(_rev_reactions_rex, '', regex=True)
        mapper = dict([(k, v) for v, k in zip(A_index[wherrev], fwdid) if v not in model._only_rev])

    Avar = pd.DataFrame(np.eye(n, n), index=A_index, columns=bvar.index)
    Avar_1 = Avar * -1
    Avar_1.index = Avar.index + '|lb'
    Avar.index = Avar.index + '|ub'
    A_1 = A * -1
    A_1.index = A.index + '|lb'
    A.index = A.index + '|ub'
    A = pd.concat([Avar, Avar_1, A, A_1], axis=0)

    # construct final b
    b = pd.concat([
        bvar.loc[:, 'ub'],
        -bvar.loc[:, 'lb'],
        b.loc[:, 'ub'],
        -b.loc[:, 'lb'],
    ], names='ub')

    non_nan_constraints = ~b.isna().values
    b = b.loc[non_nan_constraints]
    A = A.loc[non_nan_constraints]
    b.index = A.index

    fluxes_id = model.labelling_fluxes_id
    if coordinates == 'thermo':
        fluxes_id = fluxes_id.map(lambda x: model._only_rev[x] if x in model._only_rev else x)

    exclude = slice(None)
    fluxes_id = non_labelling_reactions.append(fluxes_id)

    A = A.loc[:, fluxes_id]
    S = S.loc[exclude, fluxes_id]
    h = h.loc[exclude]

    # row_norm = 0 happens when we have a rho_min = 0.0
    ineq_coef = np.linalg.norm(A, axis=1) != 0.0
    A = A.loc[ineq_coef, :]

    A.sort_index(axis=0, inplace=True)  # readability
    b = b.loc[A.index]

    eq_coef = np.linalg.norm(S, axis=1) != 0.0
    S = S.loc[eq_coef, :]
    h = h.loc[eq_coef]

    if A.index.duplicated().any():
        print(A.index[A.index.duplicated()])
        raise ValueError('constraints need to have unique names!')

    # cosmetics
    A[A == -0.0] = 0.0
    b[b == -0.0] = 0.0

    if coordinates == 'thermo':
        S.loc[:, mapper.keys()] = 0.0
        S.rename(mapper, axis=1, inplace=True)
        A.rename(mapper, axis=1, inplace=True)
        mapper = dict([(_rev_reactions_rex.sub('', k), v) for k, v in mapper.items()])

    objective = {}
    objective_expression = model.solver.objective.expression
    coefficients = objective_expression.as_coefficients_dict()
    # from cobra.util import linear_reaction_coefficients
    for var, coef in coefficients.items():
        if isinstance(var, One):
            # NB this means that no objective has been set I think...
            break
        rid = variables_id[var.name]
        if rid in A.columns:
            objective[rid] = coef

    return LabellingPolytope(A=A, b=b, S=S, h=h, mapper=mapper, objective=objective, non_labelling_reactions=non_labelling_reactions)


def rref_null_space(S: pd.DataFrame, tolerance=1e-10):
    # nsimplify changes matrix from floats to rationals, this avoids numerical issues, makes it too slow for some models
    f = functools.partial(nsimplify, **{
        'constants': (),
        'tolerance': tolerance,
        'full': False,
        'rational': True,
        'rational_conversion': 'base10',
    })
    M = Matrix(S.values).applyfunc(f)
    reduced, pivots = M.rref(simplify=True, normalize_last=True)

    free_vars = np.array([i for i in range(M.cols) if i not in pivots], dtype=np.int64)
    basis = []

    for free_var in free_vars:
        vec = [M.zero] * M.cols
        vec[free_var] = M.one
        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]
        basis.append(vec)

    NS = np.array([M._new(M.cols, 1, b) for b in basis]).astype(np.double).squeeze().T
    NS[abs(NS) < tolerance] = 0.0
    return pd.DataFrame(NS, index=S.columns, columns=S.columns[free_vars]), free_vars


def svd_null_space(S: pd.DataFrame, tolerance=1e-10):
    u, s, vh = np.linalg.svd(S.values)
    s = np.array(s.tolist())
    vh = np.array(vh.tolist())
    null_mask = s <= tolerance
    null_mask = np.append(null_mask, True)
    null_ind = np.argmax(null_mask)
    null = vh[null_ind:, :]
    freedex = [f'svd{i}' for i in range(null.shape[0])]
    return pd.DataFrame(np.transpose(null), index=S.columns, columns=freedex)


def round_polytope_keep_ellipsoid(polytope: Polytope, settings: PolyRoundSettings = PolyRoundSettings()):
    polytope = polytope.copy()
    cols = polytope.A.columns
    bool = False
    bool += np.isinf(polytope.A.values).any()
    bool += np.isinf(polytope.b.values).any()
    if bool:
        raise ValueError("Polytope assigned for rounding contains inf")

    blank_polytope = Polytope(polytope.A, polytope.b)
    MaximumVolumeEllipsoidFinder.iterative_solve(blank_polytope, settings)
    # MaximumVolumeEllipsoidFinder.iterative_solve(
    #     o_polytope, backend, hp_flags=hp_flags, verbose=verbose, sgp=sgp
    # )
    # check if the transformation is full dimensional
    _, s, _ = np.linalg.svd(blank_polytope.transformation)
    if not np.min(s) > settings.thresh / settings.accepted_tol_violation:
        raise ValueError("Rounding transformation not full dimensional")
    # check if 0 is a solution
    if not blank_polytope.b.min() > 0:
        raise ValueError("Zero point not inside rounded polytope")

    E = blank_polytope.transformation
    epsilon = blank_polytope.shift.to_frame()

    E.columns = 'R_' + cols
    E.index = cols

    polytope.apply_shift(epsilon.values)
    polytope.apply_transformation(E.values)
    E_1 = pd.DataFrame(np.linalg.inv(E), index=E.columns, columns=E.index)
    polytope.A.columns = E.columns
    polytope.transformation.columns = E.columns
    return polytope, E, E_1, epsilon


def transform_polytope_keep_transform(
    polytope: Polytope,
    settings: PolyRoundSettings = PolyRoundSettings(),
    kernel_id ='svd',
) -> Polytope:
    # PolyRoundApi.transform_polytope()
    if polytope.inequality_only:
        raise ValueError("Polytope already transformed (only contains inequality constraints)")

    polytope = polytope.copy()
    x, dist = ChebyshevFinder.chebyshev_center(polytope, settings)

    if polytope.border_distance(x) <= 0:
        raise ValueError("Chebyshev center outside polytope before transforming")

    if settings.verbose:
        print("chebyshev distance is : " + str(dist))
        pre_b_dist = polytope.border_distance(x)
        print("border distance pre-transformation is: " + str(pre_b_dist))

    if settings.verbose:
        x_0 = np.zeros(x.shape)
        b_dist_at_zero = polytope.border_distance(x_0)
        print("border distance zero-transformation is: " + str(b_dist_at_zero))

    cols = polytope.A.columns
    stoichiometry = polytope.S

    if kernel_id == 'svd':
        T = svd_null_space(stoichiometry, tolerance=settings.numerics_threshold)
        T_1 = T.T

        # put x at zero! # TODO is this correct
        polytope.apply_shift(x)
    elif kernel_id == 'rref':
        T, free_vars = rref_null_space(stoichiometry, tolerance=settings.numerics_threshold)
        T_1 = pd.DataFrame(0.0, index=T.columns, columns=T.index)
        T_1.loc[T.columns, T.columns] = np.eye(len(free_vars))

        x_star = T_1 @ x
        tau = (x - (T @ x_star))
        tau[abs(tau) < settings.numerics_threshold] = 0.0
        polytope.apply_shift(tau.values)
    else:
        raise ValueError

    tau = polytope.shift.to_frame()

    polytope.transformation.columns = cols
    polytope.apply_transformation(T)
    polytope.A.columns = T.columns
    if settings.verbose:
        u = np.zeros((T.shape[1], 1))
        norm_check = np.linalg.norm(np.matmul(stoichiometry.values, T))
        print("norm of the null space is: " + str(norm_check))
        b_dist = polytope.border_distance(u)
        print("border distance after transformation is: " + str(b_dist))
        # test if we can reproduce the original x
        trans_x = polytope.back_transform(u)
        x_rec_diff = np.max(trans_x - np.squeeze(tau.values))
        print("the deviation of the back transform is: " + str(x_rec_diff))
    if isinstance(polytope, LabellingPolytope):
        polytope._mapper = None
        polytope._objective = None
    return polytope, T, T_1, tau


# prof2 = line_profiler.LineProfiler()
# class PolytopeSamplingModel(UniformSamplingModel):
class PolytopeSamplingModel(object):
    # combine stuff from labelling polytope and mapping things
    # this one is meant to be used by
    # TODO: it is better to refactor this into a class for the sampling model and a class for the ball and cylinder
    def __init__(
            self,
            polytope: LabellingPolytope,
            pr_verbose = False,
            kernel_id ='svd',
            linalg: LinAlg = None,
            **kwargs
    ):
        if kernel_id not in ['rref', 'svd']:
            raise ValueError(f'{kernel_id} not a valid basis kernel basis')

        if polytope.A.columns.str.contains(_xch_reactions_rex).any() or \
                polytope.A.columns.str.contains(_rev_reactions_rex).any():
            print(
                'This is not a net-polytope, you sure about this? '
                'Volume computation should not include xch fluxes and is unreliable for polytope over '
                '20 dimensions such as labelling polytopes due to the algorithm implementation'
            )

        self._pr_settings = PolyRoundSettings(**{'verbose': pr_verbose, **kwargs})
        self._ker_id = kernel_id # if transform_type in ['svd', 'rref']

        normalize = kernel_id != 'rref'
        F_simp = polytope
        if F_simp.S is not None:
            F_simp = PolyRoundApi.simplify_polytope(polytope, settings=self._pr_settings, normalize=normalize)
        F_trans = LabellingPolytope.from_Polytope(F_simp, polytope)
        if F_simp.S is not None:
            F_trans, self._T, self._T_1, self._tau = transform_polytope_keep_transform(
                F_simp, self._pr_settings, kernel_id
            )
        else:
            self._T = pd.DataFrame(np.eye(F_simp.A.shape[1]), index=F_simp.A.columns, columns=F_simp.A.columns)
            self._T_1 = self._T.copy()
            self._tau = np.zeros(F_simp.A.shape[1])

        # self._Ft = LabellingPolytope.from_Polytope(F_trans, polytope)

        F_round, self._E, self._E_1, self._epsilon = round_polytope_keep_ellipsoid(F_trans, self._pr_settings)
        self._F_round = LabellingPolytope.from_Polytope(F_round)
        self._log_det_E = np.log(np.linalg.eig(self._E)[0]).sum()

        self._rounded_id = self._E_1.index
        self._transformed_id = self._T_1.index

        self._reaction_ids = polytope.A.columns.tolist()

        if linalg == None:
            linalg = LinAlg(backend='numpy')

        self._la = linalg
        new = self.to_linalg(linalg)
        self.__dict__.update(new.__dict__)

    @property
    def log_det_E(self):
        return self._log_det_E

    @property
    def kernel_id(self):
        return self._ker_id

    @property
    def dimensionality(self) -> int:
        return self._G.shape[1]

    def map_fluxes_2_rounded(self, net_fluxes: pd.DataFrame, pandalize=False):
        index = None
        if isinstance(net_fluxes, pd.DataFrame):
            index = net_fluxes.index
            net_fluxes = self._la.get_tensor(values=net_fluxes.loc[:, self.reaction_ids].values)

        transformed = self._la.tensormul_T(self._T_1, net_fluxes - self._tau.T)
        rounded = self._la.tensormul_T(self._E_1, transformed - self._epsilon.T)
        if pandalize:
            rounded = pd.DataFrame(self._la.tonp(rounded), index=index, columns=self._rounded_id)
            rounded.index.name = 'samples_id'
        return rounded

    def map_rounded_2_fluxes(self, rounded: pd.DataFrame, pandalize=False):
        index = None
        if isinstance(rounded, pd.DataFrame):
            index = rounded.index
            rounded = self._la.get_tensor(values=rounded.loc[:, self._rounded_id].values)

        # same as _to_fluxes_transform
        # transformed = self._la.tensormul_T(self._E, rounded) + self._epsilon
        # fluxes = self._la.tensormul_T(self._T, transformed) + self._tau.T

        A, b = self._to_fluxes_transform
        fluxes = self._la.tensormul_T(A, rounded) + b.T

        if pandalize:
            fluxes = pd.DataFrame(self._la.tonp(fluxes), index=index, columns=self.reaction_ids)
            fluxes.index.name = 'samples_id'
        return fluxes

    def get_initial_points(self, num_points: int):
        # UniformSamplingModel.get_initial_points(self, num_points)
        distances = self._h / self._la.norm(self._G, ord=2, axis=1)  # the arguments are ord and axis
        radius = distances.min()
        # Sample random directions and scale them to a random length inside the hypersphere.
        # self._la.sample_hypersphere()  # TODO!
        samples = self._la.randu(shape=(self.dimensionality, num_points))
        length = self._la.randu(shape=(1, num_points)) ** (1 / self.dimensionality) / self._la.norm(samples, 2, 0)
        samples = samples * self._la.diag(length) * radius
        return samples.T

    @property
    def reaction_ids(self):
        """Gets the IDs of the reactions in the model."""
        return self._reaction_ids

    def to_linalg(self, linalg: LinAlg):
        new = copy.copy(self)
        new._la = linalg
        new._G = linalg.get_tensor(values=new._F_round.A.values)
        new._h = linalg.get_tensor(values=new._F_round.b.values[:, np.newaxis])
        new._to_fluxes_transform = (
            linalg.get_tensor(values=new._F_round.transformation.values),
            linalg.get_tensor(values=new._F_round.shift.values[:, np.newaxis]),
        )
        for kwarg in ['_T', '_T_1', '_tau', '_E', '_E_1', '_epsilon']:
            value = new.__dict__[kwarg]
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                value = value.values
            new.__dict__[kwarg] = linalg.get_tensor(values=value)
        return new


class FluxCoordinateMapper(object):
    def __init__(
            self,
            model: 'LabellingModel',
            pr_verbose = False,
            kernel_id ='svd',  # basis for null-space of simplified polytope
            linalg: LinAlg = None,
            **kwargs
    ):
        # this is if we rebuild model and set new free reactions
        # free_reaction_id = [] if free_reaction_id is None else list(free_reaction_id)
        if len(model._labelling_reactions) == 0:
            raise ValueError('build the model first')

        self._la = linalg if linalg else model._la

        self._F  = extract_labelling_polytope(model, 'labelling')
        self._Ft = extract_labelling_polytope(model, 'thermo')
        self._Fn = thermo_2_net_polytope(self._Ft, pr_verbose)
        self._n_lr = len(self.labelling_fluxes_id)

        self._sampler = PolytopeSamplingModel(self._Fn, pr_verbose, kernel_id, self._la, **kwargs)

        self._fwd_id = pd.Index(self._Ft.mapper.keys())
        self._only_rev = model._only_rev
        self._fwd_idx = self._la.get_tensor(
            values=np.array([self._Ft.A.columns.get_loc(rid) for rid in self._fwd_id]),
            dtype=np.int64
        )
        self._rev_idx = self._la.get_tensor(
            values=np.array([self._Ft.A.columns.get_loc(rid) for rid in self._Ft.mapper.values()]),
            dtype=np.int64
        )
        self._only_rev_idx = self._la.get_tensor(
            values=np.array([self._F.A.columns.get_loc(rid) for rid in self._only_rev.keys()]),
            dtype=np.int64
        )
        self._nx = len(self._fwd_id)
        self._rho_bounds = self._la.zeros((self._nx, 2))
        for i, rid in enumerate(self._fwd_id):
            reaction = model.labelling_reactions.get_by_id(rid)
            self._rho_bounds[i, 0] = reaction.rho_min
            self._rho_bounds[i, 1] = reaction.rho_max

        self._samples_id = self._la._batch_size
        self._J_lt = self._la.get_tensor(shape=(0,))
        self._J_tt = self._la.get_tensor(shape=(0,))

    @property
    def samples_id(self):
        if isinstance(self._samples_id, int):
            return pd.RangeIndex(stop=self._samples_id)
        return self._samples_id.copy()

    @property
    def fwd_id(self):
        return self._fwd_id

    def net_theta_id(self, coordinate_id='rounded', hemi=False):
        if coordinate_id == 'transformed':
            return self._sampler._transformed_id.rename('net_theta_id')
        elif coordinate_id == 'rounded':
            return self._sampler._rounded_id.rename('net_theta_id')
        elif coordinate_id == 'ball':
            basis_str = 'B' if not hemi else 'HB'
            return pd.Index(
                [f'{basis_str}_{self._sampler._ker_id}_{i}' for i in range(self._sampler.dimensionality)] +
                ['R'], name='net_theta_id'
            )
        elif coordinate_id == 'cylinder':
            basis_str = 'C' if not hemi else 'HC'
            return pd.Index(
                ['phi'] +
                [f'{basis_str}_{self._sampler._ker_id}_{i}' for i in range(self._sampler.dimensionality - 2)] +
                ['R'], name='net_theta_id'
            )
        else:
            raise ValueError(f'{coordinate_id}')

    def xch_theta_id(self, log_xch=False):
        if not log_xch:
            return (self._fwd_id + '_xch').rename(name='xch_theta_id')
        return ('L_' + self._fwd_id + '_xch').rename(name='xch_theta_id')

    def theta_id(self, coordinate_id='rounded', hemi=False, log_xch=False):
        net_theta_id = self.net_theta_id(coordinate_id, hemi)
        xch_theta_id = self.xch_theta_id(log_xch)
        return net_theta_id.append(xch_theta_id).rename('theta_id')

    @property
    def fluxes_id(self):
        return self._F.A.columns.copy()

    @property
    def labelling_fluxes_id(self):
        return self.fluxes_id[len(self._F.non_labelling_reactions):]

    @property
    def thermo_fluxes_id(self):
        return self._Ft.A.columns.copy()

    def _map_theta_2_tokens(self):
        raise NotImplementedError('this is to prepare the data for transformers like in Simformer (https://arxiv.org/abs/2404.09636)')

    def _map_ball_2_polar(self, ball, pandalize=False):
        # polar coordinates is another option...
        raise NotImplementedError

    def _map_rounded_2_ball(self, rounded, hemi: bool=False, alpha_root: float=None, pandalize=False):
        if rounded.shape[-1] < 2:
            raise ValueError('only works for systems with at least 2 free dimensions!')
        index = None
        if isinstance(rounded, pd.DataFrame):
            index = rounded.index
            rounded = self._la.get_tensor(values=rounded.loc[:, self._rounded_id].values)

        norm = self._la.norm(rounded, 2, -1, keepdims=True)
        directions = rounded / norm

        if hemi:
            # this makes sure we sample on the half-sphere!
            signs = self._la.sign(directions[..., [0]])
            directions = directions * signs

        allpha = self._sampler._h.T / self._la.tensormul_T(self._sampler._G, directions)
        alpha_max = self._la.min_pos_max_neg(allpha, return_what=0 if hemi else 1, keepdims=True)

        if hemi:
            alpha_min, alpha_max = alpha_max
            first_el = directions[..., [0]]
            alpha = (rounded[..., [0]] - (alpha_min * first_el)) / first_el
            alpha_frac = alpha / (alpha_max - alpha_min)
        else:
            alpha_frac = norm / alpha_max  # fraction of max distance from polytope boundary

        if alpha_root is None:
            alpha_root = self._sampler.dimensionality

        alpha_frac = self._la.float_power(alpha_frac, alpha_root)

        result = self._la.cat([directions, alpha_frac], dim=-1)

        if pandalize:
            result = pd.DataFrame(self._la.tonp(result), index=index, columns=self.net_theta_id('ball', hemi))
            result.index.name = 'samples_id'
        return result

    def _map_ball_2_rounded(self, ball, hemi: bool=False, alpha_root: float=None, pandalize=False):
        index = None
        if isinstance(ball, pd.DataFrame):
            index = ball.index
            columns = self.net_theta_id
            ball = self._la.get_tensor(values=ball.loc[:, columns].values)

        directions = ball[..., :-1]
        alpha_frac = ball[..., [-1]]

        if alpha_root is None:
            alpha_root = self._sampler.dimensionality

        alpha_frac = self._la.float_power(alpha_frac, 1.0 / alpha_root)

        allpha = self._sampler._h.T / self._la.tensormul_T(self._sampler._G, directions)
        alpha_max = self._la.min_pos_max_neg(allpha, return_what=0 if hemi else 1, keepdims=True)

        if hemi:
            alpha_min, alpha_max = alpha_max
            alpha = alpha_frac * (alpha_max - alpha_min) + alpha_min  # fraction of chord
        else:
            alpha = alpha_frac * alpha_max  # fraction of max distance from polytope boundary

        rounded = directions * alpha
        if pandalize:
            rounded = pd.DataFrame(self._la.tonp(rounded), index=index, columns=self.net_theta_id(coordinate_id='rounded'))
            rounded.index.name = 'samples_id'
        return rounded

    def _map_ball_2_cylinder(self, ball, hemi=False, rescale_val=1.0, pandalize=False):
        index = None
        if isinstance(ball, pd.DataFrame):
            index = ball.index
        if ball.shape[-1] < 3:
            raise ValueError('not possible for polytopes K<3')
        output = self._la.vecopy(ball)
        for i in reversed(range(2, ball.shape[-1] - 1)):
            output[..., :i] /= self._la.sqrt(1.0 - output[..., [i]] ** 2)
        atan = self._la.arctan2(output[..., [0]], output[..., [1]])

        if rescale_val is not None:
            # this scales atan to [-1, 1]
            # atan is [0, pi] if _hemi else [-pi, pi]
            minb = 0.0 if hemi else -math.pi
            atan = -1 + 2 * (atan - minb) / (math.pi - minb)
            # R in [0, 1], so we scale to [-1, 1]
            output[..., -1] = -1 + 2 * output[..., -1]

        cylinder = self._la.cat([atan, output[..., 2:]], dim=-1)

        if (rescale_val is not None) and (rescale_val != 1):
            # scales to [-_bound, _bound]
            cylinder = -rescale_val + 2 * rescale_val * (cylinder + 1.0) / 2

        if pandalize:
            cylinder = pd.DataFrame(self._la.tonp(cylinder), index=index, columns=self.net_theta_id(coordinate_id='cylinder'))
            cylinder.index.name = 'samples_id'
        return cylinder

    def _map_cylinder_2_ball(self, cylinder, hemi=False, rescale_val=1.0, pandalize=False):
        index = None
        if isinstance(cylinder, pd.DataFrame):
            index = cylinder.index

        if (rescale_val is not None) and (rescale_val != 1):
            # scales cylinder to [-1, 1]
            cylinder = (cylinder + rescale_val) / (2 * rescale_val) * 2 - 1

        atan = cylinder[..., [0]]
        if rescale_val is not None:
            # scales atan is [0, pi] if _hemi else [-pi, pi]
            minb = 0.0 if hemi else -math.pi
            atan = (atan + 1) / 2 * (math.pi - minb) + minb
            # scales R back to [0, 1]
            cylinder[..., -1] = (cylinder[..., -1] + 1) / 2

        dim0 = self._la.sin(atan)
        dim1 = self._la.cos(atan)

        ball = self._la.cat([dim0, dim1, cylinder[..., 1:]], dim=-1)
        for i in range(2, ball.shape[-1] - 1):
            ball[..., :i] *= self._la.sqrt(1.0 - ball[..., [i]] ** 2)
        if pandalize:
            ball = pd.DataFrame(self._la.tonp(ball), index=index, columns=self.net_theta_id(coordinate_id='ball'))
            ball.index.name = 'samples_id'
        return ball

    def _map_rounded_2_theta(
            self, rounded, coordinate_id='rounded', hemi=False, alpha_root=None, rescale_val=1.0, pandalize=False
    ):
        index = None
        if isinstance(rounded, pd.DataFrame):
            index = rounded.index
            rounded = self._la.get_tensor(values=rounded.loc[:, self._sampler._rounded_id].values)

        if coordinate_id == 'rounded':
            theta = rounded
        elif coordinate_id == 'ball':
            theta = self._map_rounded_2_ball(rounded, hemi, alpha_root)
        elif coordinate_id == 'cylinder':
            ball = self._map_rounded_2_ball(rounded, hemi, alpha_root)
            theta = self._map_ball_2_cylinder(ball, hemi, rescale_val)
        elif coordinate_id == 'transformed':
            theta = self._la.tensormul_T(self._sampler._E_1, rounded - self._sampler._epsilon.T)

        if pandalize:
            theta = pd.DataFrame(self._la.tonp(theta), index=index, columns=self.net_theta_id(coordinate_id))
            theta.index.name = 'samples_id'
        return theta

    def _map_net_theta_2_fluxes(
            self, net_theta: pd.DataFrame, coordinate_id='rounded', hemi=False,
            alpha_root=None, rescale_val=1.0, pandalize=False
    ):
        index = None
        if isinstance(net_theta, pd.DataFrame):
            index = net_theta.index
            net_theta = self._la.get_tensor(values=net_theta.loc[:, self.net_theta_id].values)

        if coordinate_id == 'transformed':
            net_fluxes = self._la.tensormul_T(self._sampler._T, net_theta) + self._sampler._tau.T  # = transformed
        else:
            if coordinate_id == 'cylinder':
                net_theta = self._map_cylinder_2_ball(cylinder=net_theta, hemi=hemi, rescale_val=rescale_val) # = ball
            if coordinate_id != 'rounded':
                net_theta = self._map_ball_2_rounded(ball=net_theta, hemi=hemi, alpha_root=alpha_root)  # = rounded
            net_fluxes = self._sampler.map_rounded_2_fluxes(rounded=net_theta)  # = fluxes

        if pandalize:
            net_fluxes = pd.DataFrame(self._la.tonp(net_fluxes), index=index, columns=self._sampler.reaction_ids)
            net_fluxes.index.name = 'samples_id'
        return net_fluxes

    def _map_fluxes_2_net_theta(
            self, net_fluxes: pd.DataFrame, coordinate_id='rounded', hemi=False,
            alpha_root=None, rescale_val=1.0, pandalize=False
    ):
        index = None
        if isinstance(net_fluxes, pd.DataFrame):
            index = net_fluxes.index
            net_fluxes = self._la.get_tensor(values=net_fluxes.loc[:, self._sampler.reaction_ids].values)

        net_theta = self._la.tensormul_T(self._sampler._T_1, net_fluxes - self._sampler._tau.T)  # = transformed

        if coordinate_id != 'transformed':
            net_theta = self._la.tensormul_T(self._sampler._E_1, net_theta - self._sampler._epsilon.T) # = rounded
            if coordinate_id != 'rounded':
                net_theta = self._map_rounded_2_ball(net_theta, hemi, alpha_root)  # = ball
                if coordinate_id != 'ball':
                    net_theta = self._map_ball_2_cylinder(net_theta, hemi, rescale_val)  # = cylinder

        if pandalize:
            net_theta = pd.DataFrame(self._la.tonp(net_theta), index=index, columns=self.net_theta_id(coordinate_id))
            net_theta.index.name = 'samples_id'
        return net_theta

    def _rescale_xch(self, xch_fluxes, rescale_val=1.0, to_rescale_val=True):
        if rescale_val is None:
            raise ValueError
        if to_rescale_val:
            old_lo, old_hi = self._rho_bounds[:, 0], self._rho_bounds[:, 1]
            new_lo, new_hi = -rescale_val, rescale_val
        else:
            old_lo, old_hi = -rescale_val, rescale_val
            new_lo, new_hi = self._rho_bounds[:, 0], self._rho_bounds[:, 1]
        zero_one_scale = self._la.scale(xch_fluxes, lo=old_lo, hi=old_hi, rev=False)
        return self._la.scale(zero_one_scale, lo=new_lo, hi=new_hi, rev=True)

    def _expit_xch(self, xch_fluxes):
        return self._la.scale(
            self._la.expit(xch_fluxes), lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1], rev=True
        )

    def _logit_xch(self, xch_fluxes):
        return self._la.logit(self._la.scale(
            xch_fluxes, lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1], rev=False
        ))

    def frame_fluxes(self, labelling_fluxes: Union[pd.DataFrame, pd.Series, np.array], samples_id=None, trim=True):
        if isinstance(labelling_fluxes, pd.Series):
            # needed to have correct dimensions
            labelling_fluxes = labelling_fluxes.to_frame(name=labelling_fluxes.name).T

        if isinstance(labelling_fluxes, pd.DataFrame):
            samples_id = labelling_fluxes.index  # this means that the passed samples_id is ignored!
            labelling_fluxes = self._la.get_tensor(values=labelling_fluxes.loc[:, self._F.A.columns].values)

        labelling_fluxes = self._la.atleast_2d(labelling_fluxes)

        if samples_id is None:
            self._samples_id = labelling_fluxes.shape[0]
        else:
            self._samples_id = pd.Index(samples_id)
            if len(samples_id) != labelling_fluxes.shape[0]:
                raise ValueError('batch-size does not match samples_id size')
            elif self._samples_id.duplicated().any():
                raise ValueError('non-unique sample ids')
        if trim:
            labelling_fluxes = labelling_fluxes[..., len(self._F.non_labelling_reactions):]
        if labelling_fluxes.shape[-1] != self._n_lr:
            raise ValueError(f'wrong shape brahh, should be {self._n_lr}, is {labelling_fluxes.shape}, maybe wrong trim?')
        return labelling_fluxes

    def compute_dgibbsr(self, thermo_fluxes: pd.DataFrame, pandalize=False):
        if self._nx == 0:
            raise ValueError('no reversible reactions!')

        index = None
        if isinstance(thermo_fluxes, pd.DataFrame):
            index = thermo_fluxes.index
            thermo_fluxes = self._la.get_tensor(values=thermo_fluxes.loc[:, self._Ft.A.columns].values)

        xch = thermo_fluxes[..., self._rev_idx]
        net = thermo_fluxes[..., self._fwd_idx]

        xch[xch == 0.0] = 1.0
        exponent = self._la.ones(net.shape)
        exponent[net < 0.0] = -1
        T = LabellingReaction.T
        R = LabellingReaction._R
        dgibbsr = R * T * self._la.log(xch) ** exponent
        if LabellingReaction._KILOJOULE:
            dgibbsr /= 1000.0
        if pandalize:
            dgibbsr = pd.DataFrame(self._la.tonp(dgibbsr), index=index, columns=self._fwd_id + '_xch')
            dgibbsr.index.name = 'samples_id'
        return dgibbsr

    def compute_xch_fluxes(self, dgibbsr: pd.DataFrame):
        if self._nx == 0:
            raise ValueError('no reversible reactions!')
        if not isinstance(dgibbsr, pd.DataFrame):
            raise ValueError('needs to be a dataframe, since we need to .loc columns')

        dgibbsr = dgibbsr.loc[:, self._fwd_id]
        if LabellingReaction._KILOJOULE:
            dgibbsr = dgibbsr * 1000

        T = LabellingReaction.T
        R = LabellingReaction._R
        exponent = np.ones(dgibbsr.shape)
        exponent[dgibbsr > 0.0] = -1.0

        xch_fluxes = np.exp(dgibbsr / (R * T)) ** exponent
        return pd.DataFrame(xch_fluxes, index=dgibbsr.index, columns=dgibbsr.columns)

    def rounded_jacobian(
            self,
            labelling_jacobian,  # this is a jacobian of state w.r.t. fluxes, we might want to differentiate further to free variables
            thermo_fluxes,
    ):
        # raise NotImplementedError
        # TODO this is a complex function that propagates the jacobian all the
        #  way from labelling jacobian to free variables jacobian
        # TODO: need to fix fwd and reverse mapping for bi-directional fluxes
        # TODO deal with the fact that cofactor fluxes are included in the coordinate mapper!!!!

        theta = self.map_fluxes_2_theta(thermo_fluxes, is_thermo=True)

        if self._J_lt is None:
            # labelling fluxes w.r.t. thermo fluxes
            n = len(self.thermo_fluxes_id)
            self._J_lt = self._la.get_tensor(shape=(thermo_fluxes.shape[0], n, n))
            self._J_lt[...] = self._la.eye(n)[None, :, :]
            if len(self._only_rev) > 0:
                self._J_lt[..., self._only_rev_idx, self._only_rev_idx] = -1.0

        if self._nx > 0:
            xch = thermo_fluxes[..., self._rev_idx]
            net = thermo_fluxes[..., self._fwd_idx]

            drev_dnet = (xch / (1.0 - xch))
            self._J_lt[..., self._fwd_idx, self._rev_idx] = drev_dnet
            self._J_lt[..., self._fwd_idx, self._fwd_idx] = drev_dnet + 1.0

            drev_dxch = net / (1.0 - xch)**2
            self._J_lt[..., self._rev_idx, self._rev_idx] = drev_dxch
            self._J_lt[..., self._rev_idx, self._fwd_idx] = drev_dxch

        if self._J_tt is None:
            # thermo fluxes w.r.t. theta
            n = 1
            if self._logxch:
                n = thermo_fluxes.shape[0]
            self._J_tt = self._la.get_tensor(shape=(n, len(self.theta_id), len(self.labelling_fluxes_id)))
            self._J_tt[:, :len(self.net_theta_id), :-self._nx] = self._mapper.to_fluxes_transform[0].T[None, :, :]
            if not self._logxch and (self._nx > 0):
                self._J_tt[:, -self._nx, -self._nx] = self._la.ones(self._nx)

        if self._logxch and (self._nx > 0):
            sigma_xch = theta[..., -self._nx:]
            s = self._la.exp(-sigma_xch)
            C = (self._rho_bounds[:, 1] - self._rho_bounds[:, 0])[None, :]
            dxch_dsigmaxch = (C * s) / (s + 1)**2
            self._J_tt[:, -self._nx:, -self._nx:] = dxch_dsigmaxch[..., None]

        return self._J_tt @ self._J_lt @ labelling_jacobian

    def map_thermo_2_fluxes(self, thermo_fluxes: pd.DataFrame, pandalize=False):
        index = None
        if isinstance(thermo_fluxes, pd.DataFrame):
            index = thermo_fluxes.index
            thermo_fluxes = self._la.get_tensor(values=thermo_fluxes.loc[:, self.thermo_fluxes_id].values)

        fluxes = self._la.vecopy(thermo_fluxes)

        if self._nx > 0:
            xch = fluxes[..., self._rev_idx]
            net = fluxes[..., self._fwd_idx]

            if hasattr(thermo_fluxes, 'requires_grad') and thermo_fluxes.requires_grad:
                xch = xch.clone()
                net = net.clone()

            abs_net = abs(net)
            rev = (abs_net * xch) / (1.0 - xch)
            fwd = rev + abs_net
            wherrev = net < 0.0
            remember = rev[wherrev]
            rev[wherrev] = fwd[wherrev]
            fwd[wherrev] = remember

            fluxes[..., self._rev_idx] = rev
            fluxes[..., self._fwd_idx] = fwd

        if len(self._only_rev) > 0:
            fluxes[..., self._only_rev_idx] *= -1
        if pandalize:
            fluxes = pd.DataFrame(self._la.tonp(fluxes), index=index, columns=self.fluxes_id)
            fluxes.index.name = 'samples_id'
        return fluxes

    def map_theta_2_fluxes(
            self, theta: pd.DataFrame, coordinate_id='rounded', log_xch=False, hemi=False,
            alpha_root=None, rescale_val=1.0, return_thermo=False, pandalize=False
    ):
        index = None
        if isinstance(theta, pd.DataFrame):
            index = theta.index
            theta = self._la.get_tensor(values=theta.loc[:, self.theta_id(coordinate_id, hemi, log_xch)].values)

        if self._nx > 0:
            net_theta = theta[..., :-self._nx]  # this selects the net-variables
            xch_fluxes = theta[..., -self._nx:]
            if log_xch:
                xch_fluxes = self._expit_xch(xch_fluxes)
            elif rescale_val is not None:
                xch_fluxes = self._rescale_xch(xch_fluxes, to_rescale_val=False)
        else:
            net_theta = theta

        thermo_fluxes = self._map_net_theta_2_fluxes(net_theta, coordinate_id, hemi, alpha_root, rescale_val)  # should be in linalg form already
        if self._nx > 0:
            thermo_fluxes = self._la.cat([thermo_fluxes, xch_fluxes], dim=-1)
        if return_thermo:
            if pandalize:
                thermo_fluxes = pd.DataFrame(self._la.tonp(thermo_fluxes), index=index, columns=self.thermo_fluxes_id)
                thermo_fluxes.index.name = 'samples_id'
            return thermo_fluxes

        labelling_fluxes = self.map_thermo_2_fluxes(thermo_fluxes, pandalize=pandalize)
        if pandalize:
            if index is not None:
                labelling_fluxes.index = index
            labelling_fluxes.index.name = 'samples_id'
        return labelling_fluxes

    def map_fluxes_2_thermo(self, fluxes: pd.DataFrame, pandalize=False):
        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self._F.A.columns].values)

        thermo_fluxes = self._la.vecopy(fluxes)

        if len(self._only_rev) > 0:
            thermo_fluxes[..., self._only_rev_idx] *= -1

        if self._nx > 0:
            rev = thermo_fluxes[..., self._rev_idx]
            fwd = thermo_fluxes[..., self._fwd_idx]

            if hasattr(fluxes, 'requires_grad') and thermo_fluxes.requires_grad:
                rev = rev.clone()
                fwd = fwd.clone()

            net = fwd - rev
            xch = rev / fwd
            wherrev = net < 0.0
            xch[wherrev] = 1.0 / xch[wherrev]
            thermo_fluxes[..., self._rev_idx] = xch
            thermo_fluxes[..., self._fwd_idx] = net
        if pandalize:
            thermo_fluxes = pd.DataFrame(self._la.tonp(thermo_fluxes), index=index, columns=self.thermo_fluxes_id)
            thermo_fluxes.index.name = 'samples_id'
        return thermo_fluxes

    def map_fluxes_2_theta(
            self, fluxes: pd.DataFrame, coordinate_id='rounded', log_xch=False, hemi=False,
            alpha_root=None, rescale_val=1.0, is_thermo=True, pandalize=False
    ):

            # self, fluxes: pd.DataFrame, coordinate_id='rounded', is_thermo=False, pandalize=False):
        if coordinate_id not in ['transformed', 'rounded', 'ball', 'cylinder']:
            raise ValueError(f'{coordinate_id} not a valid basis coordinate system')

        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            if is_thermo:
                cols = self._Ft.A.columns
            else:
                cols = self._F.A.columns
            fluxes = self._la.get_tensor(values=fluxes.loc[:, cols].values)

        thermo_fluxes = fluxes
        if not is_thermo:
            thermo_fluxes = self.map_fluxes_2_thermo(thermo_fluxes)

        if self._nx > 0:
            xch_fluxes = thermo_fluxes[..., self._rev_idx]
            if log_xch:
                xch_fluxes = self._logit_xch(xch_fluxes)
            elif rescale_val is not None:
                xch_fluxes = self._rescale_xch(xch_fluxes, rescale_val, True)

            net_fluxes = thermo_fluxes[..., :-self._nx]
            net_theta = self._map_fluxes_2_net_theta(net_fluxes, coordinate_id, hemi, alpha_root, rescale_val)
            theta = self._la.cat([net_theta, xch_fluxes], dim=1)
        else:
            theta = self._map_fluxes_2_net_theta(thermo_fluxes, coordinate_id, hemi, alpha_root, rescale_val)

        if pandalize:
            theta = pd.DataFrame(self._la.tonp(theta), index=index, columns=self.theta_id(coordinate_id, hemi, log_xch))
            theta.index.name = 'samples_id'
        return theta

    def to_linalg(self, linalg: LinAlg):
        new = copy.copy(self)
        new._la = linalg
        new._sampler = self._sampler.to_linalg(linalg)
        for kwarg in ['_fwd_idx', '_rev_idx', '_only_rev_idx', '_rho_bounds',]:
            value = new.__dict__[kwarg]
            new.__dict__[kwarg] = linalg.get_tensor(values=value)
        return new


def make_theta_polytope(fcm: FluxCoordinateMapper):
    net_polytope = fcm._sampler._F_round
    if fcm._nx == 0:
        return net_polytope
    xch_id = fcm.xch_theta_id()
    xch_A = pd.DataFrame(0.0, columns=xch_id, index=net_polytope.A.index)
    A = pd.concat([net_polytope.A, xch_A], axis=1)
    ub_idx = fcm._fwd_id + '_xch|ub'
    lb_idx = fcm._fwd_id + '_xch|lb'
    A_xch = pd.DataFrame(0.0, columns=A.columns, index=ub_idx.append(lb_idx))
    A_xch.loc[ub_idx, xch_id] =  np.eye(fcm._nx)
    A_xch.loc[lb_idx, xch_id] = -np.eye(fcm._nx)
    A_xch[A_xch == -0.0] = 0.0
    bounds = fcm._la.tonp(fcm._rho_bounds)
    b_xch = pd.Series(np.concatenate([bounds[:, 1], bounds[:, 0]]), index=A_xch.index)
    return LabellingPolytope(A=pd.concat([A, A_xch], axis=0), b=pd.concat([net_polytope.b, b_xch]))


def sample_polytope(
        model: Union[PolytopeSamplingModel, LabellingPolytope],
        n: int = 2000,
        n_burn: int = 100,
        initial_points = None,
        thinning_factor = 3,
        n_chains: int = 4,
        new_initial_points=False,
        return_psm = False,
        phi: float = None,
        linalg: LinAlg = None,
        kernel_basis: str = 'svd',
        density=None,
        n_cdf=5,
        return_arviz=False,
        return_what='rounded',
):
    # TODO just use the function MCMC from sbmfi.estimate.simulator!
    r"""
    Hit and run sampler from uniform sampling points from a polytope,
    described via inequality constraints A*x<=b.

    Args:
        A: A Tensor describing inequality constraints
            so that all samples satisfy Ax<=b.
        b: A Tensor describing the inequality constraints
            so that all samples satisfy Ax<=b.
        x0: A `d`-dim Tensor representing a starting point of the chain
            satisfying the constraints.
        n: The number of resulting samples kept in the output.
        n_burn: The number of burn-in samples. The chain will produce
            n+n0 samples but the first n0 samples are not saved.
        seed: The seed for the sampler. If omitted, use a random seed.

    Returns:
        (n, d) dim Tensor containing the resulting samples.
    """

    if return_what not in ('all', 'fluxes', 'rounded'):
        raise ValueError(f'return_what: {return_what} not in [all, fluxes, rounded]')

    result = {}
    if isinstance(model, LabellingPolytope):
        model = PolytopeSamplingModel(model, kernel_id=kernel_basis, linalg=linalg)
        if return_psm:
            result['psm'] = model
        result['log_det_E'] = model.log_det_E

    if (phi is not None) and (phi < 1.0):
        raise ValueError('c`est ne pas possiblementenete')

    K = model.dimensionality

    if initial_points is not None:
        n_burn = 0

    n_per_chain = math.ceil(n / n_chains)
    n_tot = n_burn + n_per_chain * thinning_factor
    chains = model._la.get_tensor(shape=(n_per_chain, n_chains, K))  # use for PSRF computation

    if initial_points is None:
        x = model.get_initial_points(num_points=n_chains)
    else:
        x = initial_points
        if not x.shape[0] == n_chains:
            raise ValueError

    if density is not None:
        import torch
        if not isinstance(density, torch.distributions.Distribution) and (model._la.backend == 'torch'):
            raise NotImplementedError(f'sampling densities only works with torch distributions: {type(density)} '
                                      f'combined with torch LinAlg backend: {model._la.backend}')
        line_xs = model._la.get_tensor(shape=(1 + n_cdf, n_chains, K))
        log_probs = model._la.get_tensor(shape=(1 + n_cdf, n_chains))
        log_probs[0, :] = density.log_prob(x)  # ordering of the samples from the PDF does not matter for inverse sampling
        log_probs_selecta = model._la.arange(n_chains)

    biatch = min(2500, n_tot)
    for i in range(n_tot):
        # given x, the next point in the chain is x+alpha*r
        #             # it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        #             # so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        #             # b - A @ x is always >= 0, clamping for numerical tolerances

        if i % biatch == 0:
            # pre-sample samples from hypersphere
            # uniform samples from unit ball in d dims
            sphere_samples = model._la.sample_hypersphere(shape=(biatch, n_chains, K))
            # batch compute distances to all planes
            # A_dist = model._G[None, ...] @ model._la.transax(sphere_samples)
            A_dist = model._la.tensormul_T(model._G, sphere_samples)
            rands = model._la.randu((biatch, n_chains), dtype=model._G.dtype)

        sphere_sample = sphere_samples[i % biatch]
        ar = A_dist[i % biatch]
        rnd = rands[i % biatch]
        dist = model._h.T - model._la.tensormul_T(model._G, x)
        dist[dist < 0.0] = 0.0
        allpha = dist / ar

        alpha_min, alpha_max = model._la.min_pos_max_neg(allpha, return_what=0)

        if phi is not None:
            # this is ellipsoid aware sampling for volume computation, meaning that
            # we choose the next step to be in the intersection of the polytope and a ball of radius rho
            # a = 1  # length of ball(1)-vector is 1...
            b = (sphere_sample * x).sum(1) * 2
            c = (x * x).sum(1) - phi ** 2   # elements of ax**2 + bx + c = 0
            sqrt = model._la.sqrt(b ** 2 - 4 * c)

            phi_max = (-b + sqrt) / 2
            phi_min = (-b - sqrt) / 2

            alpha_max = model._la.minimum(phi_max, alpha_max)
            alpha_min = model._la.maximum(phi_min, alpha_min)

        if density is None:
            # this means we do vanilla hit-and-run with uniform proposal along the line
            alpha = alpha_min + rnd * (alpha_max - alpha_min)
            x = x + alpha[:, None] * sphere_sample
        else:
            # construct points along the line-segment and compute the empirical CDF from which we select the next step
            line_alphas = model._la.randu(shape=(n_cdf, n_chains, 1)) * (alpha_max - alpha_min)[None, :, None] + alpha_min[None, :, None]
            line_xs[1:] = x + line_alphas * rnd
            log_probs[1:, ] = density.log_prob(line_xs[1:])
            max_log_probs = model._la.max(log_probs, dim=0)

            normalized = log_probs - max_log_probs[None, :]
            probs = model._la.exp(normalized)  # TODO make sure this does not underflow!
            cdf = model._la.cumsum(probs, 0)  # empirical CDF
            cdf = cdf / cdf[-1, :]  # numbers between 0 and 1, now find the one closest to rnd to determine which sample is accepted
            accept_idx = model._la.argmin(abs(cdf - rnd[None, :]), 0, keepdim=False)  # indices of accepted samples
            log_probs[0, :] = log_probs[accept_idx, log_probs_selecta]  # set the log-probs of the current sample
            x = line_xs[accept_idx, log_probs_selecta]
            line_xs[0, :] = x  # set the log-probs of the current sample

        j = i - n_burn
        if (j > -1) & (j % thinning_factor == 0):
            k = j // thinning_factor
            chains[k] = x

    rounded_samples = model._la.view(chains, (n_chains * n_per_chain, K))[:n, :]

    if new_initial_points:
        new_points_idx = model._la.choice(n_chains, n)
        result['new_initial_points'] = rounded_samples[new_points_idx, :]

    if return_arviz:
        raise NotImplementedError
    if return_what in ('rounded', 'all'):
        result['rounded'] = rounded_samples
    if return_what in ('fluxes', 'all'):
        result['fluxes'] = model.map_rounded_2_fluxes(rounded_samples)
    return result

def compute_volume(
        model: Union[PolytopeSamplingModel, LabellingPolytope],
        n: int = -1,
        n0_multiplier: int = 5,
        thinning_factor: int = 1,
        epsilon: float = 1.0,
        enumerate_vertices: bool = False,
        return_all_ratios: bool = False,
        quadratic_program: bool = True,
        verbose: bool = False,
):
    # raise NotImplementedError('refactored stuff, look at this')
    psm = model
    if isinstance(model, LabellingPolytope):
        kernel_id = 'rref' if enumerate_vertices else 'svd'
        psm = PolytopeSamplingModel(model, kernel_id=kernel_id)

    if any([_xch_reactions_rex.search(rid) is not None for rid in psm.reaction_ids]):
        raise ValueError('This is a thermodynamic model that includes xch fluxes which lie in a hyper-rectangle, '
                         'it is much faster to compute the volume of the net polytope and the huper-rectangle separately!')

    K = psm.dimensionality

    if n < 0:  # this is taken from the paper
        n = min(int(400 * epsilon ** -2 * K * np.log(K)), 100000)

    if enumerate_vertices:
        if not ((psm.kernel_id == 'rref') and (psm.coordinate_id == 'transformed')):
            raise ValueError('only works with rref, pass polytope or rref volumemodel')
        F_trans = PolyRoundApi.simplify_polytope(psm.basis_polytope, normalize=False)
        if F_trans.S is not None:
            F_trans, _, _ = transform_polytope_keep_transform(F_trans, kernel_id='rref')
        vertices = V_representation(F_trans)
        n_vertices = vertices.shape[0]
        if verbose:
            print('vertices done')
        vertices = psm._la.get_tensor(values=vertices.values)
        rounded_vertices = psm._la.tensormul_T(psm._E_1, vertices - psm._epsilon.T)
        phis = psm._la.norm(rounded_vertices, 2, 1)
        phi_max = psm._la.max(phis)

    elif quadratic_program:
        np_psm = psm.to_linalg(linalg=LinAlg(backend='numpy'))
        K = psm.dimensionality
        cp_vars = np_psm._F_round.generate_cvxpy_LP()
        x = cp.Variable(K, name='slacks')
        v = cp_vars['v']
        ones = np.ones(K, dtype=np.double)
        problem = cp.Problem(cp.Maximize(ones @ x), constraints=[  # maximize absolute value, but this does not work...
            *cp_vars['constraints'],
             x - v <= 0.0,
            -x - v <= 0.0,
        ])
        # Q = np.eye(K, dtype=np.double)
        # problem = cp.Problem(cp.Maximize(cp.quad_form(cp_vars['v'], Q)), constraints=cp_vars['constraints'])
        # problem = cp.Problem(cp.Maximize(ones @ cp.abs(cp_vars['v'])), constraints=cp_vars['constraints'])
        raise NotImplementedError

    else:
        sampling_result = sample_polytope(
            model=psm, n=n * n0_multiplier, thinning_factor=thinning_factor, return_what='basis'
        )
        basis_samples = sampling_result['basis']
        phis = psm._la.norm(basis_samples, 2, 1)
        phi_max = psm._la.max(phis)  # this is the radius of the max ball that almost fully encloses the polytope

    beta = math.ceil(K * np.log(phi_max))
    ball_phis = np.array([np.exp(i / K) for i in range(0, beta + 1)])

    result = dict(
        log_B0_vol=np.log(np.pi ** (K / 2) / (scipy.special.gamma(K / 2 + 1))),
        K=K, N=n, phi_max=phi_max, beta=beta, log_det_E=psm.log_det_E,
    )
    if enumerate_vertices:
        result['n_vertices'] = n_vertices

    ratios = np.zeros(ball_phis.size -1 )

    for i, phi_i_1 in enumerate(ball_phis[1:]):
        samples = sample_polytope(psm, n=n, thinning_factor=thinning_factor, phi=phi_i_1, return_what='rounded')['rounded']
        sample_phis = psm._la.norm(samples, 2, 1)
        n_ball_i = (sample_phis <= ball_phis[i]).sum()
        ratios[i] = n / n_ball_i

    if return_all_ratios:
        result['ratios'] = ratios

    result['log_ratio'] = np.log(ratios).sum()
    if verbose:
        print('VOLDONE')
    return result


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    model, kwargs = spiro(
        backend='torch',
        auto_diff=False,
        batch_size=1,
        add_biomass=True,
        v2_reversible=True,
        ratios=True,
        build_simulator=True,
        add_cofactors=True,
        which_measurements=None,
        seed=2,
        measured_boundary_fluxes=('h_out',),
        which_labellings=['A', 'B'],
        include_bom=True,
        v5_reversible=False,
        n_obs=0,
        kernel_id='svd',
        L_12_omega=1.0,
        clip_min=None,
        transformation='ilr',
    )
    cyl_fcm = FluxCoordinateMapper(
        model=model,
        pr_verbose=False,
        kernel_id='svd',  # basis for null-space of simplified polytope
    )

    # fcm = FluxCoordinateMapper(model, kernel_id='rref')
    # psm = PolytopeSamplingModel(fcm._Fn, kernel_id='rref')
    # print(fast_FVA(psm._Ft))

    # polytope = model.flux_coordinate_mapper._Fn
    # F_simp = polytope
    # pr_set = model.flux_coordinate_mapper._sampler._pr_settings
    # if F_simp.S is not None:
    #     F_simp = PolyRoundApi.simplify_polytope(F_simp, settings=pr_set, normalize=False)
    # F_trans = LabellingPolytope.from_Polytope(F_simp, polytope)
    # if F_simp.S is not None:
    #     Ft, T, T_1, tau = transform_polytope_keep_transform(F_simp, pr_set, 'rref')
    # projected_fluxes = ['h_out', 'bm']
    # pol = model._fcm._sampler._F_round
    # V_rep = V_representation(pol)
    # print(V_rep)
    # settings = model._fcm._sampler._pr_settings
    # spol = PolyRoundApi.simplify_polytope(pol, settings=settings, normalize=False)
    # pol = LabellingPolytope.from_Polytope(spol, pol)
    # P = pd.DataFrame(0.0, index=projected_fluxes, columns=pol.A.columns)
    # P.loc[projected_fluxes, projected_fluxes] = np.eye(len(projected_fluxes))
    # pp = rref_and_project(pol, P, settings, number_type='float')
    # print(P)


    # model, kwargs = spiro(backend='numpy', v2_reversible=True, v5_reversible=False, build_simulator=False, which_measurements=None)
    # fcm = FluxCoordinateMapper(
    #     model,
    #     kernel_id='rref',
    #     logit_xch_fluxes=False,
    #     coordinate_id='transformed',
    #     pr_verbose=False,
    #     hemi_sphere=False,
    #     symmetric_rescale_val=2.0,
    # )
    # # polytope = fcm._sampler._F_round
    # # polytope = thermo_2_net_polytope(polytope)
    # #
    # # #
    # # # A,b = H_representation(v_rep)
    # #
    # # np_arr = np.hstack([polytope.b.values[:, None], -polytope.A.values])
    # # # P = _cdd_mat_pol(ineq, None, True, number_type, cdd.RepType.INEQUALITY)
    # # # g = _cdd_dual(P)
    # # # V = _cdd_mat_ar(g)
    # # print(np_arr)
    # # print(np_arr.dtype)
    # # matrix = cdd.matrix_from_array(np_arr, rep_type=cdd.RepType.INEQUALITY)
    # # print(matrix)
    # # polyhedron = cdd.polyhedron_from_matrix(matrix)
    # # matrix = cdd.copy_output(polyhedron)
    # # V = np.array(matrix.array).astype(float)
    # # vertices = []
    # # for i in range(V.shape[0]):
    # #     if V[i, 0] != 1:  # 1 = vertex, 0 = ray
    # #         raise Exception("Polyhedron is not a polytope")
    # #     elif i not in matrix.lin_set:
    # #         vertices.append(V[i, 1:])
    #
    # # vertices = pd.DataFrame(vertices, columns=polytope.A.columns)
    #
    # v_rep = V_representation(fcm._sampler._F_trans, number_type='float')
    # A, b = H_representation(v_rep)




    # import numpy as np
    # from fractions import Fraction
    # import cdd
    #
    # np_arr = np.array([[7.249999945E-01, 1.371176147E-02, -1.360247711E-01, -7.119931354E-01, 0],
    #                    [7.250000055E-01, -1.371176147E-02, 1.360247711E-01, 7.119931354E-01, 0],
    #                    [1.498749967E+00, 4.937720011E-01, -1.349053111E+00, 4.271958812E-01, 0],
    #                    [4.496250042E+00, -2.492627158E+00, 7.697439067E-01, 2.847972552E-01, -3.650953109E+00],
    #                    [4.263750037E+00, 4.837167589E-01, 1.856347346E+00, 3.559965672E-01, 1.825476554E+00],
    #                    [2.248125002E+00, 1.734143892E+00, 1.430667973E+00, 0, 0],
    #                    [3.746875071E+00, -2.739513184E+00, 1.444270450E+00, 7.119931354E-02, 0],
    #                    [2.248125033E+00, -1.246313605E+00, 3.848719418E-01, 1.423986266E-01, 1.825476554E+00]])
    #
    # # fractionater = np.vectorize(lambda x: Fraction(x))
    # # np_arr = fractionater(np_arr)
    # mat = cdd.matrix_from_array(np_arr, rep_type=cdd.RepType.INEQUALITY)

    # print(cdd.polyhedron_from_matrix(mat))

    # import numpy as np
    # from fractions import Fraction
    # import cdd.gmp as cdd_gmp
    #
    # np_arr = np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]])
    # fractionater = np.vectorize(lambda x: Fraction(x))
    # np_arr = fractionater(np_arr)
    # print(cdd_gmp.matrix_from_array(np_arr, rep_type=cdd_gmp.RepType.INEQUALITY))
    # from importlib.metadata import version
    #
    # print(version("pycddlib"))

    # sampler = fcm._sampler
    # res = sample_polytope(sampler, n=10, n_burn=0, n_chains=5, n_cdf=6, return_what='rounded')
    # rounded = res['rounded']
    # print(pd.DataFrame(rounded, columns=sampler.basis_id))
    # ball = sampler._map_rounded_2_ball(rounded, pandalize=False)
    # print(ball)
    # ball = sampler._map_ball_2_rounded(ball, pandalize=False)
    # print(ball)

    # up = UniformNetPrior(fcm, cache_size=10)
    # s = up.sample((10, ))
    # df = fcm.map_theta_2_fluxes(s, pandalize=True)
    # df = up.sample_pandalize(15)
    # psm = fcm._sampler
    # ball = sample_polytope(psm, n=3, n_burn=0, thinning_factor=1, return_what='basis')['basis']
    # psm._map_ball_2_rounded(ball)
    # cyl = psm._map_ball_2_cylinder(ball)
    # ball = psm._map_cylinder_2_ball(cyl)
