import numpy as np
import pandas as pd

from typing import Iterable, Union, Dict, Tuple
from itertools import product, cycle
from collections import OrderedDict
from cobra import Metabolite
from sbmfi.core.linalg import LinAlg
from sbmfi.core.model import LabellingModel, RatioMixin
from sbmfi.core.metabolite import EMU
from sbmfi.core.polytopia import FluxCoordinateMapper, rref_and_project, LabellingPolytope
from sbmfi.core.util import (
    hdf_opener_and_closer,
    _bigg_compartment_ids,
    make_multidex,
)
from sbmfi.lcmsanalysis.util import (
    build_correction_matrix,
    gen_annot_df,
    _strip_bigg_rex,
)
# from sbmfi.lcmsanalysis.zemzed import add_formulas
from sbmfi.lcmsanalysis.formula import Formula
from sbmfi.lcmsanalysis.adducts import emzed_adducts
from PolyRound.api import PolyRoundApi


class MDV_ObservationModel(object):
    """    """
    def __init__(
            self,
            model: LabellingModel,
            annotation_df: pd.DataFrame,
            transformation=None,
            **kwargs,
    ):
        self._la = model._la
        self._annotation_df = annotation_df
        self._observation_df = self.generate_observation_df(model=model, annotation_df=annotation_df)
        self._n_o = self._observation_df.shape[0]
        # self._natab = self._set_natural_abundance_correction() # TODO
        self._call_kwargs = kwargs
        self._state_id = model.state_id

        if transformation is not None:
            ilr_basis = kwargs.pop('ilr_basis', 'gram')
            transformation = MDV_LogRatioTransform(
                observation_df=self._observation_df,
                linalg=self._la,
                transformation=transformation,
                ilr_basis=ilr_basis
            )
        self._transformation = transformation
        self._n_d = len(self.observation_id)

    @property
    def transformation_id(self):
        if self._transformation is None:
            return None
        return self._transformation.transformation_id

    @property
    def observation_id(self) -> pd.Index:
        if self._transformation is not None:
            return self._transformation.transformation_id.copy()
        return self._observation_df.index.copy()

    @property
    def state_id(self) -> pd.Index:
        return self._state_id.copy()

    @property
    def annotation_df(self):
        return self._annotation_df.copy()

    @property
    def observation_df(self):
        return self._observation_df.copy()

    @staticmethod
    def generate_observation_df(model: LabellingModel, annotation_df: pd.DataFrame, verbose=False):
        columns = pd.Index(['met_id', 'formula', 'adduct_name', 'nC13'])
        assert columns.isin(annotation_df.columns).all()

        return_ids = model.state_id
        cols = []
        for i, (met_id, formula, adduct_name, nC13) in annotation_df.loc[:, columns].iterrows():
            if met_id not in model.measurements:
                if verbose:
                    print(f'{met_id} not in model.measurements')
                continue

            if adduct_name in ['M-H', 'M+H']:
                adduct_str = ''
            else:
                adduct_str = f'_{{{adduct_name}}}'
            oid = f'{met_id}{adduct_str}'  # id of the observation
            f = Formula(formula)
            met = model.measurements.get_by_id(met_id)
            if isinstance(met, EMU):
                n_C = len(met.positions)
            elif isinstance(met, Metabolite):
                n_C = met.elements['C']
            if (n_C != f['C']) and verbose:
                print(f'model measurement {met} with {n_C} carbons is different from annotated formula {formula}')
            model_return_id = f'{met_id}+{nC13}'
            state_idx = np.where(return_ids == model_return_id)[0][0]

            ion_row = emzed_adducts.loc[adduct_name]
            f = f * int(ion_row['m_multiplier']) \
                + Formula(ion_row['adduct_add']) \
                - Formula(ion_row['adduct_sub']) \
                + {'-': int(ion_row['z']) * -int(ion_row.get('sign_z', 1))}
            f = f.add_C13(nC13)
            isotope_decomposition = f.to_chnops()
            cols.append((i, met_id, oid, formula, adduct_name, nC13, isotope_decomposition, state_idx))

        obs_df = pd.DataFrame(cols, columns=[
            'index', 'met_id', 'ion_id', 'formula', 'adduct_name', 'nC13', 'isotope_decomposition', 'state_idx'
        ]).set_index(keys='index')
        obs_df.index = obs_df['ion_id'] + '+' + obs_df['nC13'].astype(str)
        obs_df.index.name = 'observation_id'
        obs_df = obs_df.drop_duplicates()  # TODO figure out a bug that replicates rows a bunch of times
        return obs_df.sort_values(by=['met_id', 'adduct_name', 'nC13'])  # sorting is essential for block-diagonal structure!

    def _set_natural_abundance_correction(self, isotope_threshold=1e-4, correction_threshold=0.001):
        if self._observation_df.empty:
            raise ValueError('first set observations G')
        indices = []
        values = []
        tot_obs = 0

        for (Mid, isotope_decomposition), df in self._observation_df.groupby(by=['met_id', 'isotope_decomposition']):
            formula = Formula(formula=isotope_decomposition)
            mat = build_correction_matrix(
                formula=formula, isotope_threshold=isotope_threshold, overall_threshold=correction_threshold
            )
            slicer = df['nC13'].values
            normalizing_cons = mat[:, 0].sum()  # NOTE: making sure that the last row/ first col sum to 1
            mat /= normalizing_cons
            mat = mat[slicer, :][:, slicer]
            index = np.nonzero(mat)
            vals = mat[index]
            values.append(vals)
            indices.append(np.array(index, dtype=np.int64).T + tot_obs)
            tot_obs += df.shape[0]
        values = np.concatenate(values)
        indices = np.concatenate(indices)

        return self._la.get_tensor(shape=(tot_obs, tot_obs), values=values, indices=indices)

    def sample_observations(self, mdv, n_obs=3, **kwargs):
        raise NotImplementedError

    def __call__(self, mdv, n_obs=3, return_obs=False, pandalize=False, clip_min=0, clip_max=None):
        index = None
        if isinstance(mdv, pd.DataFrame):
            index = mdv.index
            mdv = self._la.get_tensor(values=mdv.loc[:, self.state_id].values)

        observations = self.sample_observations(mdv, n_obs=n_obs, **self._call_kwargs)
        if self._transformation is not None:
            transformations = self._transformation(observations)

        if pandalize:
            n_mdv = mdv.shape[0]
            if index is None:
                index = pd.RangeIndex(n_mdv)
            obs_index = pd.RangeIndex(n_obs)
            index = make_multidex({i: obs_index for i in index}, 'samples_id', 'obs_i')
            observations = self._la.tonp(observations).transpose(1, 0, 2).reshape((n_mdv*n_obs, self._n_o))
            observations = pd.DataFrame(observations, index=index, columns=self.observation_df.index)
            if self._transformation is not None:
                transformations = self._la.tonp(transformations).transpose(1, 0, 2).reshape((n_mdv * n_obs, self._n_o))
                transformations = pd.DataFrame(transformations, index=index, columns=self.observation_id)

        if self._transformation is not None:
            if return_obs:
                return transformations, observations
            return transformations
        return observations


class MDV_LogRatioTransform():
    # https://www.tandfonline.com/doi/full/10.1080/03610926.2021.2014890?scroll=top&needAccess=true
    # http://www.leg.ufpr.br/lib/exe/fetch.php/pessoais:abtmartins:a_concise_guide_to_compositional_data_analysis.pdf
    def __init__(
            self,
            observation_df: pd.DataFrame,
            linalg: LinAlg,
            transformation: str = 'ilr',
            ilr_basis: str = 'gram',
    ):
        # TODO come up with some reasonable transform_id!
        # self._obsmod = mdv_observation_model

        if transformation not in ['alr', 'clr', 'ilr']:
            raise ValueError('not a valid log-ratio transoformation')

        self._observation_df = observation_df
        self._la = linalg

        n_o = observation_df.shape[0]
        self._n_t = n_o # number of transformed variables
        if transformation != 'clr':
            self._n_t = n_o - observation_df['ion_id'].unique().shape[0]

        if transformation == 'ilr':
            if ilr_basis == 'gram':
                self._ilr_basis = self._la.get_tensor(shape=(n_o, self._n_t))
                self._sumatrix  = self._la.get_tensor(shape=(n_o, n_o))
                i, j = 0, 0
                for ion_id, df in observation_df.groupby('ion_id', sort=False):
                    basis = self._gramm_schmidt_basis(df.shape[0])
                    k, l = basis.shape
                    self._ilr_basis[j: j + l, i: i + k] = basis.T
                    self._sumatrix[j: j + l, j: j + l]  = 1.0
                    i += k
                    j += l
                self._meantrix = self._sumatrix / self._la.sum(self._sumatrix, 0, keepdims=True)
            elif ilr_basis == 'random_sbp':
                # TODO make a random sequental binary partition?
                # TODO pass sequential binary partition for every ion;
                #   this would serve to see how much different bases affect convergence of learning
                #   {ion:
                #       [[  1,  1, -1],
                #        [  1, -1,  0]],
                #   }
                raise NotImplementedError
            else:
                raise ValueError
        elif transformation == 'alr':
            raise NotImplementedError('TODO, also need to implement self.transformation_id')
        elif transformation == 'clr':
            raise NotImplementedError('figure out how to prase sumatrix and meanatrix')
        else:
            raise ValueError

        self._transfunc = eval(f'self._{transformation}')
        self._inv_transfunc = eval(f'self._{transformation}_inv')
        self._transformation = transformation

    @property
    def transformation_id(self):
        if self._transformation == 'clr':
            return 'clr_' + self._observation_df.index
        counts = self._observation_df['ion_id'].value_counts() - 1 # TODO this might not preserve ordering...
        return pd.Index(
            [f'{self._transformation}_{k}_{v}' for k in counts.index for v in range(counts.loc[k])],
            name='transformation_id'
        )

    def _closure(self, mat, sumatrix=None):
        if sumatrix is None:
            # sum = self._la.sum(mat, dim=-1, keepdim=True)
            sum = self._la.sum(mat, -1, True)
        else:
            sum = mat @ sumatrix
        return mat / sum

    def _clr_inv(self, clrs, sumatrix=None):
        expclrs = self._la.exp(clrs)
        return self._closure(expclrs, sumatrix)

    def _clr(self, observations, meantrix=None):
        logobs = self._la.log(observations)
        if meantrix is None:
            mean = self._la.mean(logobs, dim=-1, keepdim=True)
        else:
            mean = logobs @ meantrix
        return logobs - mean

    def _ilr_inv(self, ilrs):
        return self._clr_inv(ilrs @ self._ilr_basis.T, sumatrix=self._sumatrix)

    def _ilr(self, observations):
        return self._clr(observations, meantrix=self._meantrix) @ self._ilr_basis

    def _alr_inv(self, alrs):
        raise NotImplementedError

    def _alr(self, observations):
        raise NotImplementedError

    def _gramm_schmidt_basis(self, n):
        if n == 1:
            return self._la.get_tensor(shape=(1, 1))
        basis = self._la.get_tensor(shape=(n, n - 1))
        for j in range(n - 1):
            i = j + 1
            e = self._la.get_tensor(
                values=np.array([(1 / i)] * i + [-1] + [0] * (n - i - 1), dtype=np.double)
            ) * np.sqrt(i / (i + 1))
            basis[:, j] = e
        return basis.T

    def _sbp_basis(self, sbp):
        n_pos = (sbp == 1).sum(axis=1)
        n_neg = (sbp == -1).sum(axis=1)
        psi = np.zeros(sbp.shape)
        for i in range(0, sbp.shape[0]):
            psi[i, :] = sbp[i, :] * np.sqrt((n_neg[i] / n_pos[i]) ** sbp[i, :] / np.sum(np.abs(sbp[i, :])))
        return self._clr_inv(psi)

    def inv(self, transform):
        return self._inv_transfunc(transform)

    def __call__(self, observations):
        return self._transfunc(observations)


class _BlockDiagGaussian(object):
    """convenience class to set a bunch of indices and compute observation"""
    def __init__(self, linalg: LinAlg, observation_df: pd.DataFrame):
        self._la = linalg
        self._observation_df = observation_df
        self._no = observation_df.shape[0]

        sigma_indices = []
        tot_features = 0

        self._ionindices = {}  # for setting total intensity
        for denomi, ((model_id, ion_id, ion), df) in enumerate(
                self._observation_df.groupby(['met_id', 'ion_id', 'adduct_name'], sort=False)
        ):
            n_idion = df.shape[0]
            indices_feature = np.arange(tot_features, n_idion + tot_features, dtype=np.int64)
            self._ionindices[ion_id] = self._la.get_tensor(values=indices_feature)
            indices_block = list(product(indices_feature, indices_feature, [denomi]))
            sigma_indices += indices_block
            tot_features += n_idion

        _indices_columns = ['Σ_row_idx', 'Σ_col_idx', 'denomi', 'mdv_idx']
        sigma_indices = np.array(sigma_indices)[:, [1, 0, 2]]
        map_feat_to_mdv = dict(zip(range(self._observation_df.shape[0]), self._observation_df['state_idx'].values))
        sigma_indices = np.concatenate(
            [sigma_indices, np.vectorize(map_feat_to_mdv.get)(sigma_indices[:, 0])[:, None]], axis=1
        ).astype(np.int64)

        # these indices are used to distribute values into a sparse block-diagonal matrix
        self._indices = self._la.get_tensor(values=sigma_indices)
        self._row = self._la.get_tensor(values=sigma_indices[:, 0])
        self._col = self._la.get_tensor(values=sigma_indices[:, 1])
        self._denom = self._la.get_tensor(values=sigma_indices[:, 2])
        self._mdv = self._la.get_tensor(values=sigma_indices[:, 3])

        # these are booleans to indicate diagonal and upper triangular indices from self._indices above
        offdiag_uptri = np.array([True if (j > denomi) else False for (denomi, j, k, l) in sigma_indices])
        diagionals = sigma_indices[:, 0] == sigma_indices[:, 1]
        self._diag = self._la.get_tensor(values=diagionals)
        self._uptri = self._la.get_tensor(values=offdiag_uptri)

        # these are used to distribute values into a vector of features
        self._numi = self._la.get_tensor(values=observation_df['state_idx'].values)
        self._denomi = self._denom[self._diag]

        denom_sum_indices = np.unique(sigma_indices[:, [2, 1]], axis=0)
        self._denom_sum = self._la.get_tensor(
            shape=(sigma_indices[:, 2].max() + 1, self._observation_df.shape[0]),
            indices=denom_sum_indices, values=np.ones(denom_sum_indices.shape[0], dtype=np.double)
        )  # needs to be distributed with either self._denomi or self._denom!

        self._sigma = self._la.get_tensor(shape=(self._la._batch_size, self._no, self._no))
        self._sigma_1 = self._la.get_tensor(shape=(self._la._batch_size, self._no, self._no))
        self._chol = None
        self._bias = self._la.get_tensor(shape=(self._la._batch_size, self._no,))


    @property
    def sigma_1(self):
        if self._sigma_1 is None:
            self._sigma_1 = self._la.pinv(self._sigma, rcond= 1e-12, hermitian=False)
        return self._sigma_1

    @staticmethod
    def construct_sigma_x(observation_df: pd.DataFrame, diagonal: pd.Series = 0.0001, corr=0.0):
        la = LinAlg(backend='numpy')
        if isinstance(diagonal, float):
            diagonal = pd.Series(diagonal, index=observation_df.index)
        elif isinstance(diagonal, pd.Series) and (len(diagonal) != observation_df.shape[0]):
            raise ValueError('wrong shape')

        diagonal = diagonal.loc[observation_df.index]
        idx = _BlockDiagGaussian(linalg=la, observation_df=observation_df)
        nf = len(diagonal)
        sigma = np.zeros((nf, nf))
        diagi = np.diag_indices(n=nf)
        variance = diagonal.values
        std = np.sqrt(variance)
        sigma[diagi] = variance / 2
        if corr > 0.0:
            sigma[idx._indices[idx._uptri, 0], idx._indices[idx._uptri, 1]] = \
                np.prod(std[idx._indices[idx._uptri, :2]], axis=1) * corr
        sigma += sigma.T
        return pd.DataFrame(sigma, index=observation_df.index, columns=observation_df.index)

    def set_sigma(self, sigma, verify=True):
        # this is mainly to set constant sigma

        if isinstance(sigma, pd.Series):
            sigma = self.construct_sigma_x(self._observation_df, diagonal=sigma)

        if isinstance(sigma, pd.DataFrame):
            sigma = self._la.get_tensor(
                values=sigma.loc[:, self._observation_df.index].loc[self._observation_df.index, :].values[None, :, :],
                squeeze=False
            )  # this throws error if wrong shape or features not represented

        if verify:
            sigma = self._la.get_tensor(values=sigma)  # making sure we have the correct type
            if not sigma.shape[-1] == sigma.shape[-2] == self._no:
                raise ValueError
            for i in range(sigma.shape[0]):

                variance = sigma[i, self._row[self._diag], self._col[self._diag]]
                std = self._la.sqrt(variance)
                offtri_cov = sigma[i, self._row[self._uptri], self._col[self._uptri]]
                corr_1 = std[self._row[self._uptri]] * std[self._col[self._uptri]]

                positive_var = all(variance >= 0.0)  # variance must be positive
                valid_cov = all(abs(offtri_cov) < corr_1)  # abs(correlation) <= 1
                is_diagonal = self._la.allclose(sigma, self._la.transax(sigma), rtol=1e-10)  # sigma must be diagonal
                if not (positive_var and valid_cov and is_diagonal):
                    raise ValueError

        self._sigma = sigma
        self._chol = self._la.cholesky(self._sigma)  # NB fails if not invertible!
        self._sigma_1 = None

    def compute_observations(self, s, select=True):
        # can take both simulations (full MDVs) or observation that need to be renormalized
        # s.shape = (n_simulations, n_mdv | n_observation)
        # select = True is used when passing MDVs, select = False is used when passing intensities
        observations_num = s
        if select:
            observations_num = s[..., self._numi]
        observations_denom = self._denom_sum @ self._la.transax(observations_num)
        observations_denom[observations_denom == 0.0] = 1.0
        return observations_num / self._la.transax(observations_denom)[..., self._denomi]

    def sample_sigma(self, shape=(1, )):
        noise = self._la.randn((*shape, self._no,1))
        res =  (self._la.unsqueeze(self._chol, 1) @ noise).squeeze(-1)
        return res


class ClassicalObservationModel(MDV_ObservationModel, _BlockDiagGaussian):
    # TODO incorporate natural abundance!

    def __init__(
            self,
            model: Union[LabellingModel, RatioMixin],
            annotation_df: pd.DataFrame,
            sigma_df: pd.DataFrame = None,
            transformation = None,
            clip_min=750.0,
            clip_max=None,
            normalize=True,
    ):

        kwargs = dict(clip_min=clip_min, clip_max=clip_max, normalize=normalize)
        MDV_ObservationModel.__init__(self, model, annotation_df, transformation, **kwargs)
        _BlockDiagGaussian.__init__(self, linalg=self._la, observation_df=self._observation_df)
        self._initialize_J_xs()

        # variables needed for dealing with a singular FIM
        self._permutations = {}
        self._selector = None

        if sigma_df is not None:
            self.set_sigma(sigma_df, verify=True)

    @staticmethod
    def build_models(
            model,
            annotation_dfs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
            normalize=False,
            transformation=None,
            clip_min=0.0,
            clip_max=None,
    ) -> OrderedDict:
        obsims = OrderedDict()
        for labelling_id, (annotation_df, sigma_df) in annotation_dfs.items():
            obsim = None
            if annotation_df is not None:
                obsim = ClassicalObservationModel(
                    model,
                    annotation_df=annotation_df,
                    sigma_df=sigma_df,
                    normalize=normalize,
                    transformation=transformation,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                if sigma_df is None:
                    sigma = _BlockDiagGaussian.construct_sigma_x(obsim.observation_df)
                    obsim.set_sigma(sigma, verify=True)
            obsims[labelling_id] = obsim
        return obsims

    def set_sigma_x(self, sigma_x: pd.DataFrame):
        self.set_sigma(sigma=sigma_x, verify=True)

    def _initialize_J_xs(self):
        num_sum_indices = []
        num_sum_values = []
        for i, (rowi, coli) in enumerate(zip(self._row, self._col)):
            all_denom = self._row[self._col == rowi]
            if rowi == coli:
                indices = all_denom[all_denom != rowi]
                if indices.size:
                    num_sum_indices.append([*zip(cycle([i]), self._la.tonp(indices))])
                    num_sum_values.extend([1] * indices.shape[0])
            else:
                num_sum_indices.append([[i, self._la.tonp(coli)]])
                num_sum_values.append(-1)
        num_sum_indices = np.vstack(num_sum_indices)
        self._num_sum = self._la.get_tensor(
            shape=(self._row.shape[0], self._observation_df.shape[0]),
            indices=num_sum_indices,
            values=np.array(num_sum_values, dtype=np.double)
        )  # does not need to be distributed
        self._J_xs = self._la.get_tensor(shape=(self._la._batch_size, len(self._state_id), self._n_o))

    def J_xs(self, mdv):
        mdv = self._la.atleast_2d(mdv)
        num = mdv[..., self._numi]
        denom = self._denom_sum @ num.T
        if self._la._auto_diff:
            return self._la.diff(inputs=mdv, outputs=num / denom[self._denomi])
        jac_num = self._num_sum @ num.T
        self._J_xs[:, self._mdv, self._col] = (jac_num / (denom ** 2)[self._denom]).T
        return self._J_xs

    def J_xv(self, mdv, J_sv=None, fluxes=None):
        if self._la._auto_diff:
            if fluxes is None:
                raise ValueError('fluxes are the ones that are used to generate the passed mdv')
            observation = self.compute_observations(s=mdv, select=True)
            # this assumes that mdv has been generated with the _fluxes currently set
            return self._la.diff(inputs=fluxes, outputs=observation)
        J_xs = self.J_xs(mdv=mdv)
        return J_sv @ J_xs

    def _random_selector_permutation(self, fullrank, rank):
        while True:
            permutation = self._la.randperm(n=fullrank)[:rank]
            permtup = tuple(permutation)
            if permtup not in self._permutations:
                self._permutations[permtup] = 0.0  # store determinants for every permutation
                selector = self._la.vecopy(self._selector)
                selector[permutation] = True
                yield selector

    def sigma_v(self, mdv, J_sv, rtol=1e-10, n_tries=500):
        raise NotImplementedError('needs to be able to deal with jacobians towards different flux-bases!')
        if self._selector is None:
            self._selector = self._la.get_tensor(values=np.zeros(len(model._fcm.theta_id), dtype=np.bool_))
        J_xv = self.J_xv(mdv=mdv, J_sv=J_sv)
        FIM = (J_xv @ self._la.unsqueeze(self.sigma_1, 0) @ self._la.transax(J_xv)).squeeze(0)  # Fisher Information Matrix
        sigma_v = self._la.get_tensor(shape=FIM.shape)
        summary_v = self._la.get_tensor(shape=(mdv.shape[0], 3))
        for i in range(mdv.shape[0]):
            # TODO make this actually batched computation (difficult for different ranks in same batch...)
            invertible = False
            U, S, V = self._la.svd(A=FIM[i], full_matrices=True)
            fullrank = S.shape[0]
            # numerical rank TODO maybe refine rank determination by looking at relative eigvals, filtering small/largest
            rank_FIM = sum(S > max(S) * fullrank * rtol)

            j = 0

            if rank_FIM == fullrank:
                rank = rank_FIM
                selector = self._la.vecopy(self._selector)
                selector[:] = True
                invertible = True
            else:
                self._permutations = {}
                selector_generator = self._random_selector_permutation(fullrank=fullrank, rank=rank_FIM)

            while not invertible and not (j > n_tries):
                selector = next(selector_generator)
                FIM_act = FIM[i, selector, :][:, selector]
                U, S, V = self._la.svd(A=FIM_act, full_matrices=True)
                rank_act = S.shape[0]
                rank = sum(S > max(S) * rank_act * rtol)
                if rank_act == rank:
                    # TODO we now break as soon as we find a valid invertible matrix
                    #  should make an effort to look for a minimum determinant combination??
                    invertible = True
                j += 1
            if invertible:
                mask = selector[:, None] & selector[None, :]
                sigma_v[i, mask] = (V.T @ self._la.diag(1.0 / S) @ U.T).flatten()
                summary_v[i, 0] = rank / fullrank
                summary_v[i, 1] = self._la.trace(sigma_v[i])
                summary_v[i, 2] = 1.0 / self._la.prod(S)  # |sigma_v| = 1.0 / |FIM|
        return sigma_v, summary_v

    def sample_observations(self, mdv, n_obs=3, clip_min=0.0, clip_max=None, normalize=True, **kwargs):

        # truncate will restrict out to the domain [0, ∞)
        # normalize will make sure that the Σ out = 1
        if self._chol is None:
            raise ValueError('set sigma_x')
        mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
        observations = self.compute_observations(s=mdv, select=True)  # batch x n_observables
        if n_obs == 0:  # this means we return the 'mean'
            return observations

        noise = self.sample_sigma(shape=(mdv.shape[0], n_obs))
        noisy_observations = observations[:, None, ...] + noise

        if (clip_min is not None) or (clip_max is not None):
            noisy_observations = self._la.clip(noisy_observations, clip_min, clip_max)

        if normalize:
            if not clip_min >= 0.0:
                raise ValueError('cannot normalize if we hav negative values')
            noisy_observations = self.compute_observations(noisy_observations, select=False)  # n_obs x batch x features
        return noisy_observations

    def log_lik(self, x_meas, mdv, return_observation=False):
        x_meas = self._la.atleast_2d(x_meas)  # shape = n_obs x n_mdv
        mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
        mu_o = self.compute_observations(s=mdv, select=True)  # batch x n_d
        diff = mu_o[:, None, :] - x_meas[:, None, :]  # shape = n_obs x batch x n_d
        log_lik = -0.5 * ((diff @ self.sigma_1) * diff).sum(-1)
        if return_observation:
            return log_lik, mu_o
        return log_lik


class TOF6546Alaa5minParameters(object):
    # TODO fit 2 lines and sample uniformely between them!
    def __init__(self, is_diagonal=False, has_bias=False, probabilistic=False):
        self._la = None
        self._is_diagonal = is_diagonal
        self._has_bias = has_bias
        self._probabilistic = probabilistic

        self._popt_cvhi = [-1.15955235,  1.76738371]
        self._popt_cvlo = [-1.69165038,  5.12481632]
        self._popt_corr = [-0.12690790,  0.00185103,  0.04830099,  0.92988777, -0.08980002, -1.63842671]

    def _exp_decay(self, I, lambada, y0):
        return self._la.exp(I * lambada) * y0

    def _quadratic_surface(self, I_arr, a, b, c, d, e, f):
        x = I_arr[..., 0]
        y = I_arr[..., 1]
        return  a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

    def CV(self, I):
        hi = self._exp_decay(I, *self._popt_cvhi)
        if self._probabilistic:
            randu = self._la.randu(shape=I.shape)
            lo = self._exp_decay(I, *self._popt_cvlo)
            return randu * (hi - lo) + lo  # randomly sampled CV
        return hi  # deterministic CV

    def std(self, I):
        CV = self.CV(I=I)
        return I * CV

    def var(self, I):
        std = self.std(I=I)
        return std**2

    def corr(self, I_arr):
        # make sure that x >= y, as assumed in the correlation surface
        switch = I_arr[..., 0] < I_arr[..., 1]
        memberberry = I_arr[..., 0][switch]
        I_arr[..., 0][switch] = I_arr[..., 1][switch]
        I_arr[..., 1][switch] = memberberry
        # print((I_arr[:, :, 0] > I_arr[:, :, 1]).all())

        corr = self._quadratic_surface(I_arr, *self._popt_corr)
        corr[corr < 0.0] = 0.0
        if self._probabilistic:
            noise = (self._la.randu(corr.shape) - 0.5) * 0.1
            corr += noise
        return corr

    def cov(self, I_arr, std):
        corr = self.corr(I_arr=I_arr)
        return std[..., 0] * std[..., 1] * corr

    def bias(self, I):
        raise NotImplementedError


class LCMS_ObservationModel(MDV_ObservationModel, _BlockDiagGaussian):
    def __init__(
            self,
            model: Union[LabellingModel, RatioMixin],
            annotation_df: pd.DataFrame,
            total_intensities: pd.Series,
            parameters = TOF6546Alaa5minParameters(),
            transformation = None,
            clip_min=750.0,
            clip_max=None,
    ):
        kwargs = dict(clip_min=clip_min, clip_max=clip_max)
        MDV_ObservationModel.__init__(self, model, annotation_df, transformation, **kwargs)
        _BlockDiagGaussian.__init__(self, linalg=self._la, observation_df=self._observation_df)
        self._total_intensities = {}
        self._totI = self._la.get_tensor(shape=(self._n_o,))
        self._logtotI = self._la.get_tensor(shape=(self._n_o,))
        self._p = parameters
        self._p._la = self._la
        self.set_total_intensities(total_intensities=total_intensities)

    @staticmethod
    def build_models(
            model,
            annotation_dfs: Dict[str, pd.DataFrame],
            total_intensities: pd.Series = None,
            parameters: TOF6546Alaa5minParameters = TOF6546Alaa5minParameters(),
            clip_min=750.0,
            transformation='ilr',
    ) -> OrderedDict:
        obsims = OrderedDict()
        for labelling_id, annotation_df in annotation_dfs.items():
            obsim = None
            if annotation_df is not None:
                obsim = LCMS_ObservationModel(
                    model,
                    annotation_df=annotation_df,
                    total_intensities=total_intensities,
                    parameters=parameters,
                    clip_min=clip_min,
                    transformation=transformation
                )
            obsims[labelling_id] = obsim
        return obsims

    @property
    def total_intensities(self):
        toti = OrderedDict()
        for ion_id, indices in self._ionindices.items():
            toti[ion_id] = self._la.tonp(self._totI[indices][0])
        return pd.Series(toti, name='total_intensities')

    def construct_sigma(self, logI):
        if self._p is None:
            raise ValueError('set parameters to compute sigma elements')

        sigma = self._la.get_tensor(shape=(logI.shape[0], self._n_o, self._n_o), squeeze=False)

        std = self._p.std(I=logI) / 2
        sigma[:, self._indices[self._diag, 0], self._indices[self._diag, 1]] = std**2 / 2

        if not self._p._is_diagonal:
            cov = self._p.cov(I_arr=logI[:, self._indices[self._uptri, :2]], std=std[:, self._indices[self._uptri, :2]])
            sigma[:, self._indices[self._uptri, 0], self._indices[self._uptri, 1]] = cov

        sigma += self._la.vecopy(self._la.transax(sigma))  # easiest way to make it diagonal
        return sigma

    def set_total_intensities(self, total_intensities: pd.Series):
        if total_intensities.index.duplicated().any():
            raise ValueError('double metabolites')
        for ion_id, value in total_intensities.items():
            indices = self._ionindices.get(ion_id)
            if indices is not None:
                self._totI[indices] = value
        if (self._totI <= 0.0).any():
            raise ValueError(f'a total intensity is not set: {self.total_intensities}')
        self._logtotI = self._la.log10(self._totI)  # will fail if there are any 0s left

    def sample_observations(self, mdv, n_obs=3, clip_min=750.0, clip_max=None, **kwargs):
        if self._total_intensities is None:
            raise ValueError(f'set total intensities')

        mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
        observations = self.compute_observations(s=mdv, select=True)  # batch x n_observables
        if n_obs == 0:  # this means we return the 'mean'
            return observations

        # TODO multiply with natural abundance!
        logobs = self._la.log10(observations + 1.0)
        logI = logobs + self._logtotI[None, :]  # in log space, multiplication is addition
        noisy_observations = self._la.tile(logI[:, None, :], (1, n_obs, 1))
        sigma_x = self.construct_sigma(logI=logI)
        self.set_sigma(sigma=sigma_x, verify=False)  # TODO deal with singular matrices here
        noisy_observations += self.sample_sigma(shape=(n_obs, ))

        if self._p._has_bias:
            bias = self._p.bias(I=logI)
            noisy_observations += bias

        noisy_observations = 10 ** noisy_observations
        if (clip_min is not None) or (clip_max is not None):
            # account for the fact that low intensity signals are set to `clip_min` in our emzed pipeline
            noisy_observations = self._la.clip(noisy_observations, clip_min, clip_max)

        # recompute the partial MDVs (always on simplex!)
        noisy_observations = self.compute_observations(noisy_observations, select=False)  # n_obs x batch x features
        return noisy_observations

    # def distance(self, x_meas, mdv, n_obs=3, return_observation=False):
    #     x_meas = self._la.atleast_2d(x_meas)  # shape = n_obs x n_mdv
    #     mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
    #     # mu_o = self.compute_observations(s=mdv, select=True)  # batch x n_d
    #     sim_obs = self.sample_observations(mdv, n_obs)
    #     print(sim_obs.shape, x_meas.shape)
    #     diff = sim_obs[:, None, :] - x_meas[:, None, :]  # shape = n_obs x batch x n_d


class BoundaryObservationModel(object):
    def __init__(
            self,
            model: LabellingModel,
            measured_boundary_fluxes: Iterable,
            biomass_id: str = None,  # 'bm', 'BIOMASS_Ecoli_core_w_GAM'
            check_noise_support: bool = False,
            number_type='float',
    ):
        self._la = model._la
        self._call_kwargs = {}
        self._fcm = model._fcm
        boundary_rxns = (self._fcm._Fn.S >= 0.0).all(0) | (self._fcm._Fn.S <= 0.0).all(0)
        self._bound_id = pd.Index(measured_boundary_fluxes)

        boundary_ids = self._fcm._Fn.S.columns[boundary_rxns]
        if biomass_id is not None:
            if not (biomass_id in self._fcm.fluxes_id):
                raise ValueError
            boundary_ids = boundary_ids.union([biomass_id])
        if not self._bound_id.isin(boundary_ids).all():
            raise ValueError('can only handle boundary fluxes and biomass for this observation model')

        n = len(self._bound_id)
        self._check = check_noise_support
        self._boundary_pol = None
        if check_noise_support:
            pol = self._fcm._Fn
            settings = self._fcm._sampler._pr_settings
            spol = PolyRoundApi.simplify_polytope(pol, settings=settings, normalize=False)
            pol = LabellingPolytope.from_Polytope(spol, pol)
            P = pd.DataFrame(0.0, index=self._bound_id, columns=pol.A.columns)
            P.loc[self._bound_id, self._bound_id] = np.eye(n)
            self._boundary_pol = rref_and_project(
                pol, P=P, number_type=number_type, settings=self._fcm._sampler._pr_settings
            )
            self._A = self._la.get_tensor(values=self._boundary_pol.A.values)
            self._b = self._la.get_tensor(values=self._boundary_pol.b.values)[:, None]

    @property
    def boundary_id(self):
        return self._bound_id.copy()

    def sample_observation(self, mu_bo, n_obs=1, **kwargs):
        raise NotImplementedError

    def log_lik(self, bo_meas, mu_bo):
        raise NotImplementedError

    def __call__(self, mu_bo, n_obs=1):
        # vape = boundary_fluxes.shape
        # flat = boundary_fluxes.view(vape[:-1].numel(), vape[-1])
        return self.sample_observation(mu_bo, n_obs, **self._call_kwargs)


class MVN_BoundaryObservationModel(BoundaryObservationModel):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
            measured_boundary_fluxes: Iterable,
            biomass_id: str = None,  # 'bm', 'BIOMASS_Ecoli_core_w_GAM'
            check_noise_support: bool = False,
            number_type='float',
            sigma_o=None,
            biomass_std=0.01,
            boundary_std=0.2,
    ):
        super(MVN_BoundaryObservationModel, self).__init__(
            fcm, measured_boundary_fluxes, biomass_id, check_noise_support, number_type
        )
        n = len(self._bound_id)
        if sigma_o is None:
            sigma_o = np.eye(n) * boundary_std
            if biomass_id is not None:
                bm_idx = self._bound_id.get_loc(biomass_id)
                sigma_o[bm_idx, bm_idx] = biomass_std
        self._sigma_o = self._la.get_tensor(values=sigma_o)
        self._sigma_o_1 = self._la.pinv(self._sigma_o, rcond= 1e-12, hermitian=False)

    def sample_observation(self, mu_bo, n_obs=1, **kwargs):
        if n_obs == 0:  # consistent with the MDV observation models!
            return mu_bo
        n, n_b = mu_bo.shape
        mu_bo = mu_bo[:, None, :]
        if not self._check:
            noise = self._la.randn(shape=(n, n_obs, len(self._bound_id))) @ self._sigma_o
            return abs(mu_bo + noise)  # .squeeze(0)

        output = self._la.get_tensor(shape=(n, n_obs, n_b))
        for i in range(n):
            mean = mu_bo[i, 0, :]
            j = 0
            rounds = 0
            while j < n_obs:
                noise = self._la.randn(shape=(n_obs * 5, len(self._bound_id))) @ self._sigma_o
                samples = mean + noise

                valid = self._la.transax((self._A @ self._la.transax(samples) <= self._b)).all(-1)
                k = min((j + min(valid.sum(), n_obs)), n_obs)
                if k > j:
                    output[i, j:k, :] = samples[valid][: (k - j), :]
                    j = k
                rounds += 1
                if rounds > 20:
                    raise ValueError('distribution samples outside of support')
        return output

    def log_lik(self, bo_meas, mu_bo):
        mu_bo = self._la.atleast_2d(mu_bo)  # shape = batch x n_bo
        bo_meas = self._la.atleast_2d(bo_meas)  # shape = n_obs x n_bo
        # diff = mu_bo[None, :, :] - bo_meas[:, None, :]  # shape = n_obs x batch x n_bo
        diff = mu_bo[:, None, :] - bo_meas[:, None, :]  # shape = batch x n_obs x n_bo
        return - ((diff @ self._sigma_o_1) * diff).sum(-1)


def _process_flat_frame(mdvdff, total_intensities=None, min_signal=0.05, min_frac=0.33):
    metabolites = mdvdff.columns.str.replace('\+\d+$', '', regex=True)

    if total_intensities is None:
        # NB now we work with probability vectors
        total_intensities = pd.Series(1.0, index=metabolites.unique())

    if not metabolites.unique().isin(total_intensities.index).all():
        raise ValueError('mdvdf metabolites do not have a total signal assigned')
    totI_vals = total_intensities.loc[metabolites].values

    filter_df = mdvdff * totI_vals[None, :]
    frac_ser = (filter_df > min_signal).sum(0) / mdvdff.shape[0]
    frac_ser = frac_ser.loc[frac_ser >= min_frac]

    # make sure that we do not count mdvs with only 1 signal
    n_mdv = frac_ser.index.str.rsplit('+', expand=True).to_frame(name=['met_id', 'nC13']).set_index('met_id')
    counts = n_mdv.index.value_counts()
    keep = n_mdv.index.isin(counts[counts > 1].index)
    frac_ser = frac_ser.loc[keep]
    # indices = np.where(frac_ser.index.values[:, None] == mdvdf.columns.values[None, :])[1]
    return frac_ser


def exclude_low_massiso(
        mdvdf,
        total_intensities: pd.Series = None,
        min_frac=0.33,
        min_signal=0.05,
):
    if isinstance(mdvdf.columns, pd.MultiIndex):
        res = []
        if not mdvdf.columns.names[0] == 'labelling_id':
            raise ValueError('niks')
        for i, df in mdvdf.groupby(level=0, axis=1):
            res.append(_process_flat_frame(df.droplevel(0, axis=1), total_intensities, min_signal, min_frac))
        return pd.concat(res, keys=mdvdf.columns.get_level_values(0).unique())
    else:
        return _process_flat_frame(mdvdf, total_intensities, min_signal, min_frac)


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.models.build_models import build_e_coli_anton_glc
    from sbmfi.core.polytopia import coordinate_hit_and_run_cpp
    import pickle
    from sbmfi.estimate.priors import UniFluxPrior

    import pandas as pd

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


    model, kwargs = spiro(which_measurements=None, build_simulator=True)
    annotation_df = kwargs['annotation_df']
    fluxes = kwargs['fluxes']
    observation_df = LCMS_ObservationModel.generate_observation_df(model, annotation_df)
    total_intensities = {}
    unique_ion_ids = observation_df.drop_duplicates(subset=['ion_id'])
    for _, row in unique_ion_ids.iterrows():
        total_intensities[row['ion_id']] = annotation_df.loc[
            (annotation_df['met_id'] == row['met_id']) & (annotation_df['adduct_name'] == row['adduct_name']),
            'total_I'
        ].values[0]
    total_intensities = pd.Series(total_intensities)


    obsmod_a = LCMS_ObservationModel(model, annotation_df, total_intensities)

    obsmod_b = ClassicalObservationModel(model, annotation_df)
    sigma_x = obsmod_b.construct_sigma_x(observation_df)
    obsmod_b.set_sigma(sigma_x)

    model.set_fluxes(fluxes)
    mdv = model.cascade()

    mdv = model._la.tile(mdv.T, (4, )).T
    print(123, mdv.shape)
    x_meas = obsmod_a.sample_observations(mdv, n_obs=3)

    print(x_meas)
    # obsmod_a.distance()

    # obsmod_b.sample_observations(mdv, n_obs=3)


    bs = 2
    # model, kwargs = spiro(backend='numpy', build_simulator=True, batch_size=bs)
    # model, kwargs = build_e_coli_anton_glc(backend='numpy', build_simulator=False, batch_size=bs)
    # model.set_measurements(model.measurements.list_attr('id') + ['glc__D_e'])
    # adf = kwargs['anton']['annot_df']
    # adf.loc[65] = ['glc__D_e', 0, 'C6H12O6', 'M+H']
    # com = ClassicalObservationModel(model, adf)

    # model.build_simulator(free_reaction_id=['d_out', 'h_out', 'bm'])
    # bom = MVN_BoundaryObservationModel(model, measured_boundary_fluxes=['d_out', 'h_out', 'bm'], biomass_id='bm',
    #                                    check_noise_support=True)
    # sdf = kwargs['substrate_df']
    # prior = UniFluxPrior(model._fcm)
    # t, f = prior.sample_dataframes(bs)

    # obsim = ClassicalObservationModel(model, kwargs['annotation_df'])
    # sigma = _BlockDiagGaussian.construct_sigma_x(obsim.observation_df)
    # obsim.set_sigma(sigma)
    # model.set_fluxes(f)
    # mdv = model.cascade(pandalize=True)
    # o = obsim(mdv, n_obs=2, pandalize=True)


    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model, free_reaction_id=['d_out', 'h_out', 'bm'])
    # pickle.dump(fcm, open('spiro_fcm.p', 'wb'))
    # fcm = pickle.load(open('spiro_fcm.p', 'rb'))
    # bom = MVN_BoundaryObservationModel(fcm, biomass_id='bm', measured_boundary_fluxes=['d_out', 'h_out', 'bm'], check_noise_support=True)
    # samples = coordinate_hit_and_run(bom._boundary_pol, n=20)['fluxes']
    # aa = bom(samples.values, n_obs=5)
