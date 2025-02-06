import pandas as pd
import numpy as np
from typing import Union, List
import math
import copy
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.linalg import LinAlg
from sbmfi.core.polytopia import (
    PolytopeSamplingModel,
    extract_labelling_polytope,
    thermo_2_net_polytope,
    LabellingPolytope,
)


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
        # if len(model._labelling_reactions) == 0:
        if not model._is_built:
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


    @property
    def sampler(self):
        return self._sampler

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
        if len(self._fwd_id) == 0:
            return pd.Index(name='xch_theta_id')
        if not log_xch:
            return (self._fwd_id + '_xch').rename(name='xch_theta_id')
        return ('L_' + self._fwd_id + '_xch').rename(name='xch_theta_id')

    def theta_id(self, coordinate_id='rounded', hemi=False, log_xch=False):
        net_theta_id = self.net_theta_id(coordinate_id, hemi)
        if self._nx > 0:
            xch_theta_id = self.xch_theta_id(log_xch)
            return net_theta_id.append(xch_theta_id).rename('theta_id')
        return net_theta_id

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

    def map_ball_2_polar(self, ball, pandalize=False):
        # polar coordinates is another option...
        raise NotImplementedError
        coords = np.asarray(coords)
        n = len(coords)

        # Calculate polar angles
        angles = []
        for i in range(n - 1):
            norm = np.linalg.norm(coords[i:])
            if norm == 0:
                angle = 0
            else:
                angle = np.arccos(coords[i] / norm)
            angles.append(angle)
        return np.array(angles)

    def map_polar_2_ball(self, polar, pandalize=True):
        raise NotImplementedError
        n = len(polar) + 1
        coords = []
        angles = np.asarray(polar)
        n = len(angles) + 1

        coords = np.zeros(n)
        product = 1.0
        for i in range(n):
            if i < n - 1:
                coords[i] = product * np.cos(angles[i])
                product *= np.sin(angles[i])
            else:
                coords[i] = product
        return coords

    def map_rounded_2_ball(self, rounded, hemi: bool=False, alpha_root: float=None, pandalize=False, jacobian=False):
        if jacobian:
            raise NotImplementedError
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

    def Jacobian_rounded_ball(self, ball, hemi=False, alpha_root=None):
        index = None
        if len(ball.shape) > 2:
            raise NotImplementedError(f'{ball.shape}, should be a (n_samples x n_ball) matrix, cannot handle arbitrary batch shapes for now')
        if isinstance(ball, pd.DataFrame):
            raise NotImplementedError

        K = self.sampler.dimensionality
        J = self._la.get_tensor(shape=(*ball.shape[:-1], K, K + 1))

        directions = ball[..., :-1]
        alpha_frac = ball[..., [-1]]

        if alpha_root is None:
            alpha_root = self._sampler.dimensionality
        alpha_frac = self._la.float_power(alpha_frac, 1.0 / alpha_root)

        diags = self._la.arange(K + 1)
        # derivative \frac{\partial r^{\frac{1}{\alpha}}}{\partial r} = \frac{1}{\alpha} \cdot r^{\frac{1}{\alpha} - 1}

        allpha = self._sampler._h.T / self._la.tensormul_T(self._sampler._G, directions)
        alpha_max, active_constraints = self._la.min_pos_max_neg(
            allpha, return_what=0 if hemi else 1, keepdims=True, return_indices=True
        )

        root_correction = 1 / alpha_root * alpha_frac / ball[..., [-1]]
        J[..., :, -1] = alpha_max * ball[..., :-1] * root_correction
        J[..., diags[:-1], diags[:-1]] = 1  # make an identity matrix without extra memory

        active_G = self._sampler._G[active_constraints]
        denom = self._la.tensormul_T(active_G, directions[..., None, :])
        num = self._la.einsum("bi,bj->bij", directions, active_G.squeeze(-2))

        if hemi:
            raise NotImplementedError
            # alpha_min, alpha_max = alpha_max
            # alpha = alpha_frac * (alpha_max - alpha_min) + alpha_min  # fraction of chord
        else:
            alpha = alpha_frac * alpha_max  # fraction of max distance from polytope boundary

        rounded = directions * alpha
        J[..., :, :-1] = self._la.unsqueeze(alpha, -1) * (J[..., :, :-1] - num / denom)
        return rounded, J

    def map_ball_2_rounded(self, ball, hemi: bool=False, alpha_root: float=None, pandalize=False, jacobian=False):
        index = None
        if isinstance(ball, pd.DataFrame):
            index = ball.index
            columns = self.net_theta_id
            ball = self._la.get_tensor(values=ball.loc[:, columns].values)

        if jacobian:
            K = self.sampler.dimensionality
            diags = self._la.arange(K + 1)
            J = self._la.get_tensor(shape=(*ball.shape[:-1], K, K + 1))
            if hemi:
                raise NotImplementedError
            if len(ball.shape) > 2:
                raise NotImplementedError(f'{ball.shape}, should be a (n_samples x n_ball) matrix, cannot handle arbitrary batch shapes for now')

        directions = ball[..., :-1]
        alpha_frac = ball[..., [-1]]

        if alpha_root is None:
            alpha_root = self._sampler.dimensionality
        alpha_frac = self._la.float_power(alpha_frac, 1.0 / alpha_root)

        allpha = self._sampler._h.T / self._la.tensormul_T(self._sampler._G, directions)
        alpha_max = self._la.min_pos_max_neg(allpha, return_what=0 if hemi else 1, keepdims=True, return_indices=jacobian)

        if jacobian:
            alpha_max, active_constraints = alpha_max
            root_correction = 1 / alpha_root * alpha_frac / ball[..., [-1]]
            J[..., :, -1] = alpha_max * ball[..., :-1] * root_correction
            J[..., diags[:-1], diags[:-1]] = 1  # make an identity matrix without extra memory
            active_G = self._sampler._G[active_constraints]
            denom = self._la.tensormul_T(active_G, directions[..., None, :])
            num = self._la.einsum("bi,bj->bij", directions, active_G.squeeze(-2))

        if hemi:
            alpha_min, alpha_max = alpha_max
            alpha = alpha_frac * (alpha_max - alpha_min) + alpha_min  # fraction of chord
        else:
            alpha = alpha_frac * alpha_max  # fraction of max distance from polytope boundary
        rounded = directions * alpha

        if jacobian:
            J[..., :, :-1] = self._la.unsqueeze(alpha, -1) * (J[..., :, :-1] - num / denom)

        if pandalize:
            rounded = pd.DataFrame(self._la.tonp(rounded), index=index, columns=self.net_theta_id(coordinate_id='rounded'))
            rounded.index.name = 'samples_id'
        if jacobian:
            return rounded, J
        return rounded

    def map_ball_2_cylinder(self, ball, hemi=False, rescale_val=1.0, pandalize=False, jacobian=False):
        if jacobian:
            raise NotImplementedError
        index = None
        if isinstance(ball, pd.DataFrame):
            index = ball.index
        if ball.shape[-1] < 2:
            raise ValueError('not possible for polytopes K<2')
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

        cylinder = atan
        if ball.shape[-1] > 2:
            cylinder = self._la.cat([atan, output[..., 2:]], dim=-1)

        if (rescale_val is not None) and (rescale_val != 1):
            # scales to [-_bound, _bound]
            cylinder = -rescale_val + 2 * rescale_val * (cylinder + 1.0) / 2

        if pandalize:
            cylinder = pd.DataFrame(self._la.tonp(cylinder), index=index, columns=self.net_theta_id(coordinate_id='cylinder'))
            cylinder.index.name = 'samples_id'
        return cylinder

    def Jacobian_cylinder_polar_cylinder(self, polar_cylinder, hemi=False, rescale_val=1.0):
        index = None
        if isinstance(polar_cylinder, pd.DataFrame):
            raise NotImplementedError
            # index = polar_cylinder.index
            # columns = self.net_theta_id(coordinate_id='cylinder', hemi=hemi)
            # polar_cylinder = self._la.get_tensor(values=polar_cylinder.loc[:, columns].values)

        K = self.sampler.dimensionality
        J = self._la.get_tensor(shape=(*polar_cylinder.shape[:-1], K + 1, K))

        # correct for rescaling
        cyl_rescale = 1
        if (rescale_val is not None) and (rescale_val != 1):
            # scales cylinder to [-1, 1]
            cyl_rescale = 2 * rescale_val / 2
            polar_cylinder = (polar_cylinder + rescale_val) / (2 * rescale_val) * 2 - 1

        atan = polar_cylinder[..., [0]]
        atan_rescale = 1
        r_rescale = 1
        if rescale_val is not None:
            # y in [c, d], x in [a, b], dy/dx = (d-c)/(b-a)

            # scales atan is [0, pi] if _hemi else [-pi, pi]
            minb = 0.0 if hemi else -math.pi
            atan = (atan + 1) / 2 * (math.pi - minb) + minb
            atan_rescale = (math.pi - minb) / 2

            # scales R back to [0, 1]
            polar_cylinder[..., -1] = (polar_cylinder[..., -1] + 1) / 2
            r_rescale = 1 / 2

        diags = self._la.arange(K)
        J[..., diags + 1, diags] = 1 * cyl_rescale
        J[..., -1, -1] *= r_rescale
        sin = self._la.sin(atan)
        cos = self._la.cos(atan)
        J[..., [0], 0] = cos * cyl_rescale * atan_rescale
        J[..., [1], 0] = -sin * cyl_rescale * atan_rescale

        cylinder = [sin, cos, polar_cylinder[..., 1:]]
        cylinder = self._la.cat(cylinder, dim=-1)
        return cylinder, J

    def Jacobian_ball_cylinder(self, cylinder):
        index = None
        if isinstance(cylinder, pd.DataFrame):
            raise NotImplementedError

        K = self.sampler.dimensionality
        J = self._la.get_tensor(shape=(*cylinder.shape[:-1], K + 1, K + 1))

        ball = self._la.vecopy(cylinder)
        diags = self._la.arange(K + 1)

        J[..., diags, diags] = 1

        for i in range(2, ball.shape[-1] - 1):
            sqrt_1_r2 = self._la.sqrt(1.0 - ball[..., [i]] ** 2)
            J[..., :i, i] = ball[..., :i] * ball[..., [i]] / sqrt_1_r2
            J[..., diags[:i], diags[:i]] *= sqrt_1_r2
            ball[..., :i] *= sqrt_1_r2

        # print(self._la.norm(ball[:, :K], axis=1))
        return ball, J

    def map_cylinder_2_ball(self, cylinder, hemi=False, rescale_val=1.0, pandalize=False, jacobian=False):
        index = None
        if isinstance(cylinder, pd.DataFrame):
            index = cylinder.index
            columns = self.net_theta_id(coordinate_id='cylinder', hemi=hemi)
            cylinder = self._la.get_tensor(values=cylinder.loc[:, columns].values)

        cylinder = cylinder + 0.0  # copy data, since we modify in-place

        if jacobian:
            K = self.sampler.dimensionality
            J_polar_rescale = self._la.get_tensor(shape=(*cylinder.shape[:-1], K + 1, K))
            J_ball_cylinder = self._la.get_tensor(shape=(*cylinder.shape[:-1], K + 1, K + 1))
            diags = self._la.arange(K + 1)
            J_ball_cylinder[..., diags, diags] = 1

            cyl_rescale = 1
            atan_rescale = 1
            r_rescale = 1

        if (rescale_val is not None) and (rescale_val != 1):
            # scales cylinder to [-1, 1]
            cyl_rescale = 2 * rescale_val / 2
            cylinder = (cylinder + rescale_val) / (2 * rescale_val) * 2 - 1

        atan = cylinder[..., [0]]
        if rescale_val is not None:
            # y in [c, d], x in [a, b], dy/dx = (d-c)/(b-a)

            # scales atan is [0, pi] if _hemi else [-pi, pi]
            minb = 0.0 if hemi else -math.pi
            atan = (atan + 1) / 2 * (math.pi - minb) + minb
            atan_rescale = (math.pi - minb) / 2

            # scales R back to [0, 1]
            cylinder[..., -1] = (cylinder[..., -1] + 1) / 2
            r_rescale = 1 / 2

        sin = self._la.sin(atan)
        cos = self._la.cos(atan)
        if jacobian:
            J_polar_rescale[..., diags[:-1] + 1, diags[:-1]] = 1 * cyl_rescale  # all the cylinder coordinates and radius
            J_polar_rescale[..., -1, -1] *= r_rescale  # rescaling radius r
            J_polar_rescale[..., [0], 0] = cos * cyl_rescale * atan_rescale  # rescaling atan and d x_1 \ d theta = d sin(theta) / d theta = cos(theta)
            J_polar_rescale[..., [1], 0] = -sin * cyl_rescale * atan_rescale  # rescaling atan and d x_2 \ d theta = d cos(theta) / d theta = -sin(theta)

        ball = self._la.cat([sin, cos, cylinder[..., 1:]], dim=-1)
        for i in range(2, ball.shape[-1] - 1):
            sqrt_1_r2 = self._la.sqrt(1.0 - ball[..., [i]] ** 2)
            if jacobian:
                J_ball_cylinder[..., :i, i] = ball[..., :i] * ball[..., [i]] / sqrt_1_r2
                J_ball_cylinder[..., diags[:i], diags[:i]] *= sqrt_1_r2
            ball[..., :i] *= sqrt_1_r2

        if pandalize:
            ball = pd.DataFrame(self._la.tonp(ball), index=index, columns=self.net_theta_id(coordinate_id='ball'))
            ball.index.name = 'samples_id'
        if jacobian:
            J = self._la.tensormul_T(self._la.transax(J_polar_rescale), J_ball_cylinder)
            return ball, J
        return ball

    def map_rounded_2_net_theta(
            self, rounded, coordinate_id='rounded', hemi=False, alpha_root=None, rescale_val=1.0, pandalize=False
    ):
        index = None
        if isinstance(rounded, pd.DataFrame):
            index = rounded.index
            rounded = self._la.get_tensor(values=rounded.loc[:, self._sampler._rounded_id].values)

        if coordinate_id == 'rounded':
            theta = rounded
        elif coordinate_id == 'ball':
            theta = self.map_rounded_2_ball(rounded, hemi, alpha_root)
        elif coordinate_id == 'cylinder':
            ball = self.map_rounded_2_ball(rounded, hemi, alpha_root)
            theta = self.map_ball_2_cylinder(ball, hemi, rescale_val)
        elif coordinate_id == 'transformed':
            theta = self._la.tensormul_T(self._sampler._E_1, rounded - self._sampler._epsilon.T)

        if pandalize:
            theta = pd.DataFrame(self._la.tonp(theta), index=index, columns=self.net_theta_id(coordinate_id))
            theta.index.name = 'samples_id'
        return theta

    def map_net_theta_2_fluxes(
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
                net_theta = self.map_cylinder_2_ball(cylinder=net_theta, hemi=hemi, rescale_val=rescale_val) # = ball
            if coordinate_id != 'rounded':
                net_theta = self.map_ball_2_rounded(ball=net_theta, hemi=hemi, alpha_root=alpha_root)  # = rounded
            net_fluxes = self._sampler.map_rounded_2_fluxes(rounded=net_theta)  # = fluxes

        if pandalize:
            net_fluxes = pd.DataFrame(self._la.tonp(net_fluxes), index=index, columns=self._sampler.reaction_ids)
            net_fluxes.index.name = 'samples_id'
        return net_fluxes

    def map_fluxes_2_net_theta(
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
                net_theta = self.map_rounded_2_ball(net_theta, hemi, alpha_root)  # = ball
                if coordinate_id != 'ball':
                    net_theta = self.map_ball_2_cylinder(net_theta, hemi, rescale_val)  # = cylinder

        if pandalize:
            net_theta = pd.DataFrame(self._la.tonp(net_theta), index=index, columns=self.net_theta_id(coordinate_id))
            net_theta.index.name = 'samples_id'
        return net_theta

    def rescale_xch(self, xch_fluxes, rescale_val=1.0, to_rescale_val=True):
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

    def expit_xch(self, xch_fluxes):
        return self._la.scale(
            self._la.expit(xch_fluxes), lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1], rev=True
        )

    def logit_xch(self, xch_fluxes):
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
            alpha_root=None, rescale_val=1.0, return_thermo=True, pandalize=False
    ):
        index = None
        if isinstance(theta, pd.DataFrame):
            index = theta.index
            theta = self._la.get_tensor(values=theta.loc[:, self.theta_id(coordinate_id, hemi, log_xch)].values)
        else:
            theta = self._la.vecopy(theta)  # theta is modified in-place in some map function, so we need to copy here
        if self._nx > 0:
            net_theta = theta[..., :-self._nx]  # this selects the net-variables
            xch_fluxes = theta[..., -self._nx:]
            if log_xch:
                xch_fluxes = self.expit_xch(xch_fluxes)
            elif rescale_val is not None:
                xch_fluxes = self.rescale_xch(xch_fluxes, to_rescale_val=False)
        else:
            net_theta = theta

        thermo_fluxes = self.map_net_theta_2_fluxes(net_theta, coordinate_id, hemi, alpha_root, rescale_val)  # should be in linalg form already
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
                xch_fluxes = self.logit_xch(xch_fluxes)
            elif rescale_val is not None:
                xch_fluxes = self.rescale_xch(xch_fluxes, rescale_val, True)

            net_fluxes = thermo_fluxes[..., :-self._nx]
            net_theta = self.map_fluxes_2_net_theta(net_fluxes, coordinate_id, hemi, alpha_root, rescale_val)
            theta = self._la.cat([net_theta, xch_fluxes], dim=1)
        else:
            theta = self.map_fluxes_2_net_theta(thermo_fluxes, coordinate_id, hemi, alpha_root, rescale_val)

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


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.core.polytopia import sample_polytope


    def log_prob_rev(self, z, context=None):
        ball, J_bc = self._fcm.map_cylinder_2_ball(z, rescale_val=self._rescale, jacobian=True)
        rounded, J_rb = self._fcm.map_ball_2_rounded(ball, jacobian=True)
        J = self._la.tensormul_T(J_rb, self._la.transax(J_bc))
        abs_dets = abs(self._la.det(J))

    m,k = spiro(backend='torch')
    fcm = FluxCoordinateMapper(m)
    res = sample_polytope(fcm.sampler, n=50)
    cyl_pol = fcm.map_rounded_2_net_theta(res['rounded'], coordinate_id='cylinder', pandalize=False)
    cyl, J = fcm.Jacobian_cylinder_polar_cylinder(cyl_pol)
    ball, J2 = fcm.Jacobian_ball_cylinder(cyl)
    r1, J3 = fcm.Jacobian_rounded_ball(cyl)
    ball, J = fcm.map_cylinder_2_ball(cyl_pol, jacobian=True)
    rounded, J4 = fcm.map_ball_2_rounded(cyl, jacobian=True)

    print(r1)
    print(rounded)