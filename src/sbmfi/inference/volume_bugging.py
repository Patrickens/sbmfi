from sbmfi.core.polytopia import LabellingPolytope, PolytopeSamplingModel, compute_polytope_halfspaces, \
    fast_FVA
from sbmfi.estimate.priors import sampling_tasks
from sbmfi.legacy.hr_sampler import volume_tasks, volume_worker
import pandas as pd
import scipy
import itertools
import numpy as np
from string import ascii_lowercase
import multiprocessing as mp

def make_flux_pol_from_vertices(vertices: np.array, verbose=False, transform_type='svd', basis_coordinates='rounded'):
    cols = pd.Index(list(ascii_lowercase[:vertices.shape[1]]))

    A, b = compute_polytope_halfspaces(vertices, number_type='float')
    A[abs(A) < 1e-12] = 0.0
    A = pd.DataFrame(A, columns=cols)
    A_ub = pd.DataFrame(np.eye(vertices.shape[1]), index=cols + '|ub', columns=cols)
    A_lb = pd.DataFrame(-np.eye(vertices.shape[1]), index=cols + '|lb', columns=cols)
    A_all = pd.concat([A, A_ub, A_lb], axis=0)

    b_all = pd.Series(0.0, index=A_all.index)
    b_all.loc[A.index] = b
    b_all.loc[A_ub.index] = vertices.max(0)
    b_all.loc[A_lb.index] = -vertices.min(0)

    pol = LabellingPolytope(A=A_all, b=b_all)
    vsm = PolytopeSamplingModel(pol, pr_verbose=verbose, transform_type=transform_type,
                                basis_coordinates=basis_coordinates)
    return pol, vsm


def orthantilizing(polytope):
    fva = fast_FVA(polytope)
    bidirection = fva.loc[(fva['min'] < 0.0) & (fva['max'] > 0.0)]
    orthants = pd.DataFrame(itertools.product((True, False), repeat=bidirection.shape[0]), columns=bidirection.index)
    return orthants


def make_b_constraint_df(polytope, orthants):
    forward = orthants.astype(int)
    reverse = (~orthants).astype(int)
    forward.columns += '|ub'
    reverse.columns += '|lb'

    b = polytope.b.copy()
    ub = forward * b.loc[forward.columns]
    lb = reverse * b.loc[reverse.columns]

    # NB this dataframe contains all the bounds for the b vector
    return pd.concat([lb, ub], axis=1).sort_index(axis=1)


def rejection_vol_approx(A, b, samples, n=50000):
    hi = samples.max(0)
    lo = samples.min(0)
    vol = np.prod(hi - lo)
    proposals = np.random.uniform(lo, hi, size=(n, A.shape[1]))
    m = ((A.values @ proposals.T) <= b.values[:, None]).all(0).sum()
    return (m / n) * vol


def n_gon_points(n):
    # TODO vectorize this
    points = []
    for i in range(n):
        points.append((
            np.cos(2 * i * np.pi / n),  # x
            np.sin(2 * i * np.pi / n),  # y
        ))
    return np.array(points)


def construct_random_hypercube_polytope(K=12, N_rev=8, seed=0):
    if N_rev > K:
        raise ValueError

    np.random.seed(seed)

    cols = pd.Index(list(ascii_lowercase[:K]))

    # bounds = pd.DataFrame(np.random.uniform(size=(n_dim, 2)) ** 2, index=cols, columns=['lb', 'ub']) * np.random.uniform(size=)
    bounds = pd.DataFrame(np.random.uniform(size=(K, 2)) ** 2, index=cols, columns=['lb', 'ub']) * \
             scipy.stats.loguniform.rvs(a=0.8, b=100, size=K)[:, None]
    bounds.iloc[N_rev:, 0] = 0

    A_ub = pd.DataFrame(np.eye(K), index=cols + '|ub', columns=cols)
    A_lb = pd.DataFrame(np.eye(K), index=cols + '|lb', columns=cols)
    A = pd.concat([A_ub, -A_lb]).sort_index()
    b = pd.Series(bounds.values.reshape(K * 2), index=A.index)

    polytope = LabellingPolytope(A=A, b=b)
    orthants = orthantilizing(polytope)
    b_constraint_df = make_b_constraint_df(polytope, orthants)
    b_diff_rev = b_constraint_df.loc[:, A_ub.index[:N_rev]] + b_constraint_df.loc[:, A_lb.index[:N_rev]].values
    b_diff_rev.columns = cols[:N_rev]
    b_diff_fwd = b.loc[A_ub.index[N_rev:]] + b.loc[A_lb.index[N_rev:]].values
    b_diff_fwd.index = cols[N_rev:]

    diff_df = b_diff_rev.merge(b_diff_fwd.to_frame().T, how='outer', left_index=True, right_index=True)
    diff_df.loc[:, cols[N_rev:]] = b_diff_fwd.values

    volumes = np.log(diff_df).sum(1)
    volumes.name = 'log_vol'
    orthants.index = volumes
    b_constraint_df.index = volumes

    idxs = np.argsort(volumes)[::-1]  # sorted from largest to smallest volume
    return dict(polytope=polytope, orthants=orthants.iloc[idxs], b_constraint_df=b_constraint_df.iloc[idxs])

def ratio_ball_cube(K=12, R=1):
    vol_ball = np.pi ** (K / 2) / (scipy.special.gamma(K / 2 + 1)) * R ** K
    vol_rect = (2*R) ** K
    return dict(rel_vol=vol_rect/vol_ball, rho_max=np.sqrt(K), vol_ball=vol_ball)


def make_hypercube_voldf(
        K=14,  # dimensionality
        N_rev=8,  # 2**N_rev orthants
        num_processes=4,
        N_biggest = 15,  # compute volume of N_biggest orthants
):
    res = construct_random_hypercube_polytope(K, N_rev)
    polytope = res['polytope']
    b_constraint_df = res['b_constraint_df']
    orthants = res['orthants']

    if N_biggest is not None:
        n = orthants.shape[0]
        b_constraint_df = b_constraint_df.iloc[:min(n, N_biggest)]
        orthants = orthants.iloc[:min(n, N_biggest)]

    sampling_task_generator = sampling_tasks(
        polytope, b_constraint_df=b_constraint_df, return_kwargs=True,
    )
    volume_task_generator = volume_tasks(sampling_task_generator)

    if num_processes == 0:
        result = {}
        for i, task in enumerate(volume_task_generator):
            result[i] = volume_worker(*task)
    else:
        pool = mp.Pool(num_processes)
        result = pool.starmap(volume_worker, volume_task_generator)
        pool.close()
        pool.join()

    voldf = pd.DataFrame.from_dict(result)
    if num_processes == 0:
        voldf = voldf.T

    try:
        voldf.index = orthants.index
        voldf.to_excel(f"C:\python_projects\pysumo\chapriors\hypercube_volumes_rref_K{K}_rev{N_rev}.xlsx")
    except:
        pass
    return voldf

def split_polytope_n_ways():
    # TODO make a function that splits a polytope with n hyper-planes;
    #  this way we can compute the volume of the whole and of the parts
    #  the sum of the parts should equal the whole, this could be orhtants (easiest to implement) or random hyper-planes
    pass



if __name__ == "__main__":
    voldf = make_hypercube_voldf(K=12, N_rev=6, num_processes=4, N_biggest=32)
