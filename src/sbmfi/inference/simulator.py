from multiprocessing import Pool
# from torch.multiprocessing import Pool  # TODO: make this corresponding to the backend?
import tables as pt
import warnings
import torch
import psutil
import os, pickle, io
import math
import numpy as np
import pandas as pd
import multiprocessing as mp
import contextlib
from typing import Iterable, Union, Dict, Tuple
from torch.distributions import constraints
from torch.utils.data import Dataset
from collections import OrderedDict
from scipy.spatial import ConvexHull
from functools import lru_cache
from PolyRound.api import PolyRoundApi
# from torch.distributions import Distribution  # TODO perhaps make this the base class of _BaseSimulator
from sbmfi.core.model import EMU_Model, LabellingModel, RatioMixin
from sbmfi.core.simulfuncs import (
    init_observer,
    obervervator_worker,
    observator_tasks,
)
from sbmfi.core.observation import (
    BoundaryObservationModel,
    MDV_ObservationModel,
)
from sbmfi.inference.priors import (
    _BasePrior,
    _FluxPrior,
    UniFluxPrior,
    # ThermoPrior,
    RatioPrior,
    _CannonicalPolytopeSupport,
)
from sbmfi.core.util import (
    _excel_polytope,
    hdf_opener_and_closer,
    _bigg_compartment_ids,
    make_multidex,
    profile,
)
from line_profiler import line_profiler
from sbmfi.core.polytopia import PolytopeSamplingModel, V_representation, fast_FVA, LabellingPolytope
from arviz.data.io_dict import from_dict
from arviz import InferenceData
import arviz as az
import tqdm
# logger = logging.getLogger(__name__)
# NOTE: pt.NaturalNameWarning are thrown for the column-names of the substrate_df which contain an illegal %
warnings.simplefilter('ignore', pt.NaturalNameWarning)


"""
https://andrewcharlesjones.github.io/journal/21-effective-sample-size.html
https://emcee.readthedocs.io/en/v2.2.1/user/pt/
https://people.duke.edu/~ccc14/sta-663/MCMC.html
https://python.arviz.org/en/stable/
https://pymcmc.readthedocs.io/en/latest/modelchecking.html
https://distribution-explorer.github.io/multivariate_continuous/lkj.html
https://bayesiancomputationbook.com/markdown/chp_08.html
https://michael-franke.github.io/intro-data-analysis/bayesian-p-values-model-checking.html
"""


# prof2 = line_profiler.LineProfiler()
class _BaseSimulator(object):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _FluxPrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        if not model._is_built:
            raise ValueError('need to build model')
        if not substrate_df.index.unique().all():
            raise ValueError('non-unique identifiers for labelling!')

        self._prior = prior
        if prior is not None:
            if not prior._fcm.labelling_fluxes_id.equals(model.fluxes_id):
                raise ValueError('prior has different labelling fluxes than model')
            if not model._fcm.theta_id.equals(prior.theta_id):
                raise ValueError('theta of model and prior are different')

        self._obmods = OrderedDict()
        self._obsize = {}

        has_log_prob = []
        i, j = 0, 0
        for k, (labelling_id, obmod) in enumerate(mdv_observation_models.items()):
            model.set_input_labelling(substrate_df.loc[labelling_id])  # NB check whether valid susbtrate_df
            if not model.state_id.equals(obmod.state_id):
                raise ValueError
            if not model._la == obmod._la:
                raise ValueError
            self._obmods[labelling_id] = obmod
            has_log_prob.append(hasattr(obmod, 'log_lik'))
            self._obsize[labelling_id] = i, i + obmod._n_d
            i += obmod._n_d

        self._is_exact = all(has_log_prob) & (hasattr(boundary_observation_model, 'log_lik') if
                                              boundary_observation_model is not None else True)
        self._model = model
        self._la = model._la
        self._substrate_df = substrate_df.loc[list(self._obmods.keys())]

        if boundary_observation_model is not None:
            bo_id = boundary_observation_model.boundary_id
            bo_fluxes_id = bo_id.to_series().replace({v: k for k, v in model._only_rev.items()})
            if not bo_fluxes_id.isin(model.fluxes_id).all():
                raise ValueError
            if not model._la == boundary_observation_model._la:  # TODO make sure that device also matches
                raise ValueError
            # NB this means that we always append the boundary fluxes to the end of the last dimension!
            self._bo_idx = self._la.get_tensor(  # TODO maybe make this select from prior fluxes??
                values=np.array([self._model.fluxes_id.get_loc(rid) for rid in bo_fluxes_id], dtype=np.int64)
            )
        self._bomsize = 0 if boundary_observation_model is None else len(self._bo_idx)
        self._bom = boundary_observation_model
        self._did = None
        self._x_meas = None
        self._x_meas_id = None
        self._true_theta = None
        self._true_theta_id = None

    @property
    def data_id(self):
        if self._did is None:
            did = {}
            for labelling_id, obmod in self._obmods.items():
                did[labelling_id] = obmod.observation_id
            if 'BOM' in did:
                raise ValueError
            if self._bomsize > 0:
                did['BOM'] = self._bom.boundary_id
            self._did = make_multidex(did, 'labelling_id', 'data_id')
            self._did.name = 'data_id'
        return self._did.copy()

    @property
    def theta_id(self):
        return self._model._fcm.theta_id

    def simulate(
            self,
            fluxes,
            n_obs=3,
            return_mdvs=False, # whether to return mdvs, observation_average or noisy observations
            pandalize=False,
    ):
        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self._model._fcm.fluxes_id].values)

        n_obshape = max(1, n_obs)
        slicer = 0 if n_obs == 0 else slice(None)
        n_f = fluxes.shape[0]
        self._model.set_fluxes(fluxes, index, trim=True)

        if return_mdvs:
            result = self._la.get_tensor(shape=(n_f, self._model._ns * len(self._obmods)))
        else:
            result = self._la.get_tensor(shape=(n_f, n_obshape, len(self.data_id)))
            if self._bomsize > 0:
                result[:, slicer, -self._bomsize:] = self._bom.sample_observation(
                    self._model._fluxes[:, self._bo_idx], n_obs=n_obs
                )

        i=0
        for j, (labelling_id, obmod) in enumerate(self._obmods.items()):
            self._model.set_input_labelling(input_labelling=self._substrate_df.loc[labelling_id])
            mdv = self._model.cascade()
            if return_mdvs:
                result[:, j*n_state : (j+1) * n_state] = mdv
            else:
                result[:, slicer, i:i+obmod._n_d] = obmod(mdv, n_obs=n_obs)
                i += obmod._n_d

        if pandalize:
            if return_mdvs:
                columns = make_multidex({k: self._model.state_id for k in self._obmods}, 'labelling_id', 'mdv_id')
                result = pd.DataFrame(result, index=index, columns=columns)
            else:
                result = self._la.tonp(result).transpose(1, 0, 2).reshape((n_f * n_obshape, len(self.data_id)))
                if index is None:
                    index = pd.RangeIndex(n_f)
                if n_obs > 0:
                    obs_index = pd.RangeIndex(n_obshape)
                    index = make_multidex({k: obs_index for k in index}, 'samples_id', 'obs_i')
                result = pd.DataFrame(result, index=index, columns=self.data_id)
        return result

    def set_measurement(self, x_meas: pd.Series):
        if isinstance(x_meas, pd.Series):
            name = 'measurement' if not x_meas.name else x_meas.name
            x_meas = x_meas.to_frame(name=name).T
        x_meas_index = None
        if isinstance(x_meas, pd.DataFrame):
            x_meas_index = x_meas.index
            x_meas = x_meas.loc[:, self.data_id].values
        if x_meas_index is None:
            x_meas_index = pd.RangeIndex(x_meas.shape[0])
        elif isinstance(x_meas_index, pd.MultiIndex):
            raise ValueError
        self._x_meas = self._la.get_tensor(values=x_meas)
        self._x_meas_id = x_meas_index
        if (self._bomsize > 0) and self._bom._check:
            if not self._la.transax((self._bom._A @ self._x_meas[:, -self._bomsize].T <= self._bom._b)).all():
                raise ValueError('boundary measurements are outside polytope')

    def set_true_theta(self, theta: pd.Series):
        if isinstance(theta, pd.DataFrame):
            if theta.shape[0] > 1:
                raise ValueError
            theta = theta.iloc[0]
        self._true_theta = self._la.atleast_2d(self._la.get_tensor(values=theta.loc[self._model._fcm.theta_id].values))
        self._true_theta_id = theta.name

    def simulate_true_data(self, n_obs=0, pandalize=True):
        if self._true_theta is None:
            raise ValueError('set true_theta')
        fluxes = self._model._fcm.map_theta_2_fluxes(self._true_theta)
        vv = self._la.tile(fluxes.T, (self._la._batch_size, )).T
        true_data = self.simulate(vv, n_obs, pandalize=pandalize)
        if not pandalize:
            return true_data[[0]]
        true_data = true_data.iloc[[0]]
        true_data.index = pd.RangeIndex(true_data.shape[0])
        return true_data

    @property
    def measurements(self):
        return pd.DataFrame(self._la.tonp(self._x_meas), index=self._x_meas_id, columns=self.data_id)

    @property
    def true_theta(self):
        if self._true_theta is None:
            return
        return pd.Series(self._la.tonp(self._true_theta[0]), index=self.theta_id, name=self._true_theta_id)


class DataSetSim(_BaseSimulator):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _FluxPrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
            num_processes=0,
            epsilon=1e-12,
    ):
        super(DataSetSim, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)
        self._eps = epsilon
        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        self._num_processes = num_processes
        self._mp_pool = None
        if num_processes > 0:
            self._mp_pool = self._get_mp_pool()

    def _get_mp_pool(self):
        if (self._mp_pool is None) or (hasattr(self._mp_pool, '_state') and (self._mp_pool._state == 'CLOSE')):
            self._mp_pool = mp.Pool(
                processes=self._num_processes, initializer=init_observer,
                initargs=(self._model, self._obmods, self._eps)
            )
        return self._mp_pool

    def _fill_results(self, result, worker_result):
        input_labelling = worker_result['input_labelling']
        labelling_id = input_labelling.name
        start, stop = worker_result['start_stop']
        start_stop_idx = self._la.arange(start, stop)
        i, j = self._obsize[labelling_id]
        i_obs = self._substrate_df.index.get_loc(labelling_id)
        result['validx'][start_stop_idx, i_obs] = worker_result['validx_chunk']

        for key in result.keys():
            if key == 'data':
                result['data'][start_stop_idx, :, i:j] = worker_result['data_chunk']
            elif key == 'validx':
                continue
            else:
                result[key][start_stop_idx, i_obs] = worker_result[f'{key}_chunk']

    def simulate_set(
            self,
            fluxes,
            n_obs=3,
            fluxes_per_task=None,
            what='data',
            break_i=-1,
            close_pool=True,
    ) -> OrderedDict:
        # TODO perhaps also parse samples_id
        fluxes = self._model._fcm.frame_fluxes(fluxes, trim=True)

        result = {}
        result['validx'] = self._la.get_tensor(shape=(fluxes.shape[0], len(self._obmods)), dtype=np.bool_)

        if what not in ('all', 'data', 'mdv'):
            raise ValueError('not sure what to simulate')
        if fluxes.shape[0] < self._la._batch_size:
            raise ValueError(f'n must be at least batch size: {self._la._batch_size}')

        if what != 'data':
            result['mdv'] = self._la.get_tensor(shape=(fluxes.shape[0], len(self._obmods), len(self._model.state_id)))
        if what != 'mdv':
            n_obshape = max(1, n_obs)
            result['data'] = self._la.get_tensor(shape=(fluxes.shape[0], n_obshape, len(self.data_id)))

        if self._bomsize > 0:
            slicer = 0 if n_obs == 0 else slice(None)
            bo_fluxes = fluxes[:, self._bo_idx]  # TODO only works with torch now, since we sample from a torch distribution!
            result['data'][:, slicer, -self._bomsize:] = self._bom(bo_fluxes, n_obs=n_obs)

        if fluxes_per_task is None:
            fluxes_per_task = math.ceil(fluxes.shape[0] / max(self._num_processes, 1))

        tasks = observator_tasks(
            fluxes, substrate_df=self._substrate_df, fluxes_per_task=fluxes_per_task, n_obs=n_obs, what=what
        )

        if self._num_processes == 0:
            init_observer(self._model, self._obmods, self._eps)
            for i, task in enumerate(tasks):
                worker_result = obervervator_worker(task)
                self._fill_results(result, worker_result)
                # self._fill_results(result, obervervator_worker(task))
                if (break_i > -1) and (i > break_i):
                    break
        else:
            mp_pool = self._get_mp_pool()
            for worker_result in mp_pool.imap_unordered(obervervator_worker, iterable=tasks):
                self._fill_results(result, worker_result)
            if close_pool:
                mp_pool.close()
                mp_pool.join()
        return result

    @staticmethod
    def prepare_for_sbi(
            result: OrderedDict,
            x_coordinate_sys='data',
            device='cpu',
    ):
        # TODO randomize perhaps?
        # cast to correct datatype and I guess do some other pre-processing?
        if x_coordinate_sys not in ['observation', 'transformation']:
            raise ValueError('only observation or transformation')
        x = torch.as_tensor(result[x_coordinate_sys], dtype=torch.float32, device=device)
        theta = torch.as_tensor(result['theta'], dtype=torch.float32, device=device)
        _, n_t = theta.shape
        n_samples, n_obs, n_x = x.shape
        theta = theta.unsqueeze(1).repeat((1, n_obs, 1))
        return OrderedDict([(x_coordinate_sys, x.view(n_obs * n_samples, n_x)), ('theta', theta.view(n_obs * n_samples, n_t))])

    @hdf_opener_and_closer(mode='a')
    def to_hdf(
            self,
            hdf,
            result: OrderedDict,
            dataset_id: str,
            append=True,
            expectedrows_multiplier=10,
    ):
        if 'substrate_df' not in hdf.root:
            # TODO think about storing the ilr_basis, annotation_df, total intensities and TOFPArameters to file
            # this signals that the hdf has been freshly created
            self._substrate_df.to_hdf(hdf.filename, key='substrate_df', mode=hdf.mode, format='table')
            pt.Array(hdf.root, name='mdv_id', obj=self._model.state_id.values.astype(str))
            pt.Array(hdf.root, name='fluxes_id', obj=self._model._fcm.fluxes_id.values.astype(str))  # NB these are the untrimmed fluxes
            self.data_id.to_frame(index=False).to_hdf(hdf.filename, key='data_id', mode=hdf.mode, format='table')
        else:
            # this is for existing files
            substrate_df = pd.read_hdf(hdf.filename, key='substrate_df', mode=hdf.mode)
            if not self._substrate_df.equals(substrate_df):
                raise ValueError('hdf has different substrate_df')

            for what, compare in {
                'fluxes': self._model._fcm.fluxes_id,
                'mdv': self._model.state_id,
                'data': self.data_id,
            }.items():
                if what == 'data':
                    what_id = pd.MultiIndex.from_frame(pd.read_hdf(hdf.filename, key='data_id', mode=hdf.mode))
                else:
                    what_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
                if not compare.equals(what_id):
                    raise ValueError(f'{what}_id is different between model and hdf')

        if (dataset_id in hdf.root) and not append:
            hdf.remove_node(hdf.root, name=dataset_id, recursive=True)

        if dataset_id not in hdf.root:
            hdf.create_group(hdf.root, name=dataset_id)

        dataset_children = hdf.root[dataset_id]._v_children
        if (len(dataset_children) > 0) and (dataset_children.keys() != result.keys()):
            raise ValueError(f'result {result.keys()} has different data than dataset {dataset_children.keys()}; cannot append!')

        for item, array in result.items():
            if isinstance(array, pd.DataFrame):
                array = array.values
            array = self._la.tonp(array)
            if item in hdf.root[dataset_id]:
                ptarray = hdf.root[dataset_id][item]
            else:
                atom = pt.Atom.from_type(str(array.dtype))
                ptarray = pt.EArray(
                    hdf.root[dataset_id], name=item, atom=atom, shape=(0, *array.shape[1:]),
                    expectedrows=array.shape[0] * expectedrows_multiplier, chunkshape=None,
                )
            ptarray.append(array)

    @staticmethod
    @hdf_opener_and_closer(mode='r')
    def read_hdf(
            hdf:   str,
            dataset_id: str,
            what:  str,
            labelling_id: Union[str, Iterable[str]] = None,
            start: int = None,
            stop:  int = None,
            step:  int = None,
            pandalize: bool = True,
    ) -> Union[np.array, pd.DataFrame]:
        if (dataset_id not in hdf.root):
            raise ValueError(f'{dataset_id} not in hdf')
        elif (what not in hdf.root[dataset_id]):
            raise ValueError(f'{what} not in {dataset_id}')

        xcsarr = hdf.root[dataset_id][what].read(start, stop, step) # .squeeze()  # TODO why did we squeeze before?
        validx = hdf.root[dataset_id]['validx'].read(start, stop, step)

        if not pandalize:
            return xcsarr, validx

        substrate_df = pd.read_hdf(hdf.filename, key='substrate_df', mode=hdf.mode)
        labelling_ids = substrate_df.index.rename('labelling_id')
        if labelling_id is None:
            labelling_idx = np.arange(len(labelling_ids))
        else:
            labelling_idx = labelling_ids.get_loc(labelling_id)
        labelling_id = labelling_ids[labelling_idx]

        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = xcsarr.shape[0]

        samples_id = pd.RangeIndex(start, stop, step, name='samples_id')
        validx = pd.DataFrame(validx[:, labelling_idx], index=samples_id, columns=labelling_id)

        if what == 'fluxes':
            xcs_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
            return pd.DataFrame(xcsarr, index=samples_id, columns=xcs_id), validx

        elif what == 'validx':
            return validx, validx

        elif what == 'data':
            data_id = pd.MultiIndex.from_frame(pd.read_hdf(hdf.filename, key=f'data_id', mode=hdf.mode))
            i_obs = (*range(xcsarr.shape[1]), )
            dataframes = [pd.DataFrame(xcsarr[:, i, :], index=samples_id, columns=data_id) for i in i_obs]
            dataframe = pd.concat(dataframes, keys=i_obs, names=['i_obs', 'samples_id'])
            return dataframe.swaplevel(0, 1, 0).sort_values(by=['samples_id', 'i_obs'], axis=0).loc[:, labelling_id], validx

        elif what == 'mdv':
            mdv_id = pd.Index(hdf.root['mdv_id'].read().astype(str), name='mdv_id')
            return pd.concat([
                pd.DataFrame(xcsarr[:, i], index=samples_id, columns=mdv_id)
                for i in labelling_idx], axis=1, keys=labelling_id
            ), validx

    def create_inference_data(self):
        pass


def simulate_prior_predictive(
        simulator: _BaseSimulator,
        inference_data: az.InferenceData = None,
        n=20000,
        include_prior_predictive=True,
        num_processes=2,
        n_obs=0,
):
    model = simulator._model
    prior_theta = simulator._prior.sample(sample_shape=(n, ))

    if inference_data is None:
        result = dict(theta=prior_theta[None, :, :])
    else:
        prior_dataset = az.convert_to_dataset(
            {'theta': prior_theta[None, :, :]},
            dims={'theta': ['theta_id']},
            coords={'theta_id': model._fcm.theta_id.tolist()},
        )
        inference_data.add_groups(
            group_dict={'prior': prior_dataset},
        )

    if include_prior_predictive:
        dsim = DataSetSim(
            model=model,
            substrate_df=simulator._substrate_df,
            mdv_observation_models=simulator._obmods,
            boundary_observation_model=simulator._bom,
            num_processes=num_processes,
        )
        fluxes = model._fcm.map_theta_2_fluxes(prior_theta)
        prior_data = dsim.simulate_set(fluxes, n_obs=n_obs)['data']
        dims = {'simulated_data': ['data_id']}
        coords = {'data_id': [f'{i[0]}: {i[1]}' for i in simulator.data_id.tolist()]}
        if n_obs == 0:
            prior_data = model._la.transax(prior_data, 0, 1)
        else:
            dims['simulated_data'] = ['obs_idx', 'data_id']
            prior_data = prior_data[None, :, :, :]

        if inference_data is None:
            result['simulated_data'] = prior_data
        else:
            prior_dataset = az.convert_to_dataset(
                {'simulated_data': prior_data},
                dims=dims,
                coords=coords,
            )
            inference_data.add_groups(
                group_dict={'prior_predictive': prior_dataset},
            )

    if inference_data is None:
        return result


class MCMC(_BaseSimulator):
    # TODO think about implementing MALA, this only requires the computation of gradients of log_prob

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _BasePrior,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        super(MCMC, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)

        self._fcm = model._fcm
        self._sampler = self._fcm._sampler

    def log_lik(
            self,
            fluxes,
            return_posterior_predictive=False,
            sum=True,
    ):
        if not self._is_exact:
            raise ValueError(
                'some observation models do not have a .log_prob, meaning that exact inference is impossible'
            )
        if self._x_meas is None:
            raise ValueError('set measurement')

        mu_o = self.simulate(fluxes, n_obs=0, return_mdvs=False, pandalize=False)

        n_f = fluxes.shape[0]
        n_meas = self._x_meas.shape[0]
        n_bom = 1 if self._bomsize > 0 else 0

        log_lik = self._la.get_tensor(shape=(n_f, n_meas, len(self._obmods) + n_bom))

        if self._bomsize > 0:
            bo_meas = self._x_meas[:, -self._bomsize:]
            mu_bo = mu_o[:, 0, -self._bomsize:]
            log_lik[..., -1] = self._bom.log_lik(bo_meas=bo_meas, mu_bo=mu_bo)

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            j, k = self._obsize[labelling_id]
            x_meas_o = self._x_meas[..., j:k]
            mu_o_i = mu_o[:, 0, j:k]
            ll = obmod.log_lik(x_meas_o, mu_o_i)
            log_lik[..., i] = ll

        if sum:
            log_lik = self._la.sum(log_lik, axis=(1, 2), keepdims=False)

        if return_posterior_predictive:
            return log_lik, mu_o
        return log_lik

    def log_prob(
            self,
            value,
            return_posterior_predictive=False,
            evaluate_prior=False,
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = value.shape
        viewlue = self._la.view(value, shape=(math.prod(vape[:-1]), vape[-1]))

        n_f = viewlue.shape[0]
        k = len(self._obmods) + (1 if self._bom is None else 2) # the 2 is for a column of prior and boundary probabilities
        n_meas = self._x_meas.shape[0]
        log_prob = self._la.get_tensor(shape=(n_f, n_meas, k))
        fluxes = self._fcm.map_theta_2_fluxes(viewlue)

        if evaluate_prior:
            # NB not necessary for uniform prior
            # NB this also checks support! the hr is guaranteed to sample within the support
            # NB since priors are currently torch objects, this will not work with numpy backend
            #   which has proven the faster option for the hr-sampler
            log_prob[..., -1] = self._prior.log_prob(viewlue)

        log_lik = self.log_lik(fluxes, return_posterior_predictive, False)
        if return_posterior_predictive:
            log_lik, mu_o = log_lik

        log_prob[..., :-1] = log_lik
        log_prob = self._la.view(self._la.sum(log_prob, axis=(1, 2), keepdims=False), shape=(*vape[:-1], 1))
        if return_posterior_predictive:
            return log_prob, self._la.view(mu_o, shape=(*vape[:-1], len(self._did)))
        return log_prob

    def run(
            self,
            initial_points = None,
            n: int = 2000,
            n_burn = 0,
            thinning_factor=3,
            n_chains: int = 4,
            kernel=None,
            n_cdf=5,
            algorithm='cdf',
            line_how='uniform',
            line_proposal_std=2.0,
            xch_how='gauss',
            xch_proposal_std=0.4,
            return_post_pred=False,
            evaluate_prior=False,
            kernel_kwargs=None,
    ) -> az.InferenceData:
        # TODO: this publication talks about this algo, but has a different acceptance procedure:
        #  doi:10.1080/01621459.2000.10473908
        #  doi:10.1007/BF02591694  Rinooy Kan article

        if self._fcm._sampler.basis_coordinates == 'transformed':
            raise NotImplementedError('transform the chains to transformed')

        if algorithm == 'mh':
            # TODO modify acceptance step!
            n_cdf = 1
            mh_min = self._la.get_tensor(shape=(n_chains, 1))
            accept_idx = self._la.get_tensor(shape=(n_chains, 1), dtype=np.int64)
        elif algorithm != 'cdf':
            raise ValueError

        batch_size = n_chains * n_cdf
        if (self._la._batch_size != batch_size) or not self._model._is_built:
            # this way the batch processing is corrected
            self._la._batch_size = batch_size
            self._model.build_simulator(**self._fcm.fcm_kwargs)

        K = self._sampler.dimensionality

        chains = self._la.get_tensor(shape=(n, n_chains, len(self._fcm.theta_id)))
        kernel_dist = self._la.get_tensor(shape=(n, n_chains, 1))

        sim_data = None
        if return_post_pred:
            sim_data = self._la.get_tensor(shape=(n, n_chains, len(self.data_id)))

        if initial_points is None:
            net_basis_points = self._sampler.get_initial_points(num_points=n_chains)
            theta = self._fcm.append_xch_flux_samples(net_basis_samples=net_basis_points, return_type='theta')
        else:
            theta = initial_points

        theta = self._la.tile(theta, (n_cdf, 1))  # remember that the new batch size is n_chains x n_cdf

        ker_kwargs = {'return_posterior_predictive': return_post_pred, 'evaluate_prior': evaluate_prior}
        if kernel is None:
            kernel = self.log_prob
            if kernel_kwargs is not None:
                ker_kwargs = {**ker_kwargs, **kernel_kwargs}

        line_thetas = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self._fcm.theta_id)))
        line_kernel_dist = self._la.get_tensor(shape=(1 + n_cdf, n_chains, 2)) # last dimension is prior probabilities and distances
        dist = kernel(theta, **ker_kwargs)
        if return_post_pred:
            line_data = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self.data_id)))
            dist, data = dist
        line_kernel_dist[0] = dist[:n_chains]  # ordering of the samples from the PDF does not matter for inverse sampling
        selector = self._la.arange(n_chains)

        theta = theta[: n_chains, :]
        n_rev = len(self._model._fcm._fwd_id)

        n_tot = n_burn + n * thinning_factor
        biatch = min(2500, n_tot)
        for i in tqdm.trange(n_tot, ncols=100):
            if i % biatch == 0:
                # pre-sample samples from hypersphere
                # uniform samples from unit ball in d dims
                sphere_samples = self._la.sample_hypersphere(shape=(biatch, n_chains, K))
                # batch compute distances to all planes
                ARs = self._sampler._G[None, ...] @ self._la.transax(sphere_samples)
                rands = self._la.randu((biatch, n_chains), dtype=self._sampler._G.dtype)
                # TODO: https://link.springer.com/article/10.1007/BF02591694
                #  implement coordinate hit-and-run (might be faster??)

            # given x, the next point in the chain is x+alpha*r
            #             # it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
            #             # so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
            #             # b - A @ x is always >= 0, clamping for numerical tolerances
            ar = ARs[i % biatch]
            sphere_sample = sphere_samples[i % biatch]
            rnd = rands[i % biatch]

            pol_dist = self._sampler._h - self._sampler._G @ theta[..., :K].T
            pol_dist[pol_dist < 0.0] = 0.0
            alpha_min = pol_dist / ar
            alpha_max = self._la.vecopy(alpha_min)

            alpha_max[alpha_max < 0.0] = alpha_max.max()
            alpha_max = alpha_max.min(0)
            if isinstance(alpha_max, tuple):
                alpha_max = alpha_max[0]

            alpha_min[alpha_min > 0.0] = alpha_min.min()
            alpha_min = alpha_min.max(0)
            if isinstance(alpha_min, tuple):
                alpha_min = alpha_min[0]

            # construct proposals along the line-segment and compute the empirical CDF from which we select the next step
            line_alphas = self._la.sample_bounded_distribution(
                shape=(n_cdf, ), lo=alpha_min, hi=alpha_max, which=line_how, std=line_proposal_std
            )
            net_basis_points = theta[..., :K] + line_alphas[..., None] * sphere_sample

            xch_fluxes = None
            if n_rev > 0:
                # in case there are exchange fluxes, construct them here
                current_xch = theta[..., -n_rev:]
                if self._fcm.logit_xch_fluxes:
                    current_xch = self._fcm._sigmoid_xch(current_xch)
                xch_fluxes = self._la.sample_bounded_distribution(
                    shape=(n_cdf, n_chains), lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                    mu=current_xch, which=xch_how, std=xch_proposal_std
                )

            line_thetas[1:] = self._fcm.append_xch_flux_samples(
                net_basis_samples=net_basis_points, xch_fluxes=xch_fluxes, return_type='theta'
            )
            dist = kernel(line_thetas[1:], **ker_kwargs)

            if return_post_pred:
                dist, data = dist
                line_data[1:] = data

            line_kernel_dist[1:] = dist
            max_line_probs = line_kernel_dist.max(0)
            if isinstance(max_line_probs, tuple):
                max_line_probs = max_line_probs[0]

            if algorithm =='cdf':
                normalized = line_kernel_dist - max_line_probs[None, :]
                probs = self._la.exp(normalized)  # TODO make sure this does not underflow!
                cdf = self._la.cumsum(probs, 0)  # empirical CDF
                cdf = cdf / cdf[-1, :]  # numbers between 0 and 1, now find the one closest to rnd to determine which sample is accepted
                number_picker = cdf - rnd[None, :, None]
                selector = cdf
                # make sure that we select the 'next point' instead of the closest one!
                number_picker[number_picker < 0.0] = float('inf')
                accept_idx = self._la.argmin(number_picker, 0, keepdims=False)  # indices of accepted samples
            else:
                mh_ratio = self._la.exp(self._la.minimum(mh_min, line_kernel_dist[1, ...] - line_kernel_dist[0, ...]))
                selector = mh_ratio - rnd[:, None]
                accept_idx[selector < 0] = 0
                accept_idx[selector > 0] = 1

            accepted_probs = line_kernel_dist[accept_idx[:, 0], selector]
            line_kernel_dist[0] = accepted_probs # set the log-probs of the current sample
            theta = line_thetas[accept_idx[:, 0], selector]
            line_thetas[0, ...] = theta  # set the log-probs of the current sample
            if return_post_pred:
                data = line_data[accept_idx[:, 0], selector]
                line_data[0, ...] = data
            j = i - n_burn
            if (j % thinning_factor == 0) and (j > -1):
                k = j // thinning_factor
                kernel_dist[k] = accepted_probs
                chains[k] = theta
                if return_post_pred:
                    sim_data[k] = data

        if return_post_pred:
            sim_data = {
                'simulated_data': self._la.transax(sim_data, dim0=1, dim1=0)
            }

        attrs = {
                'n_burn': n_burn,
                'thinning_factor': thinning_factor,
                'n_cdf': n_cdf,
                'line_how': line_how,
                'line_proposal_std': line_proposal_std,
                'xch_how': xch_how,
                'xch_proposal_std': xch_proposal_std,
            }
        if self.true_theta is not None:
            attrs['true_theta'] = self._la.tonp(self._true_theta)
            attrs['true_theta_id'] = self._true_theta_id

        return az.from_dict(
            posterior={
                'theta': self._la.transax(chains, dim0=1, dim1=0)  # chains x draws x param
            },
            dims={
                'theta': ['theta_id'],
                'observed_data': ['measurement_id', 'data_id'],
                'simulated_data': ['data_id'],
            },
            coords={
                'theta_id': self._model._fcm.theta_id.tolist(),
                'measurement_id': self._x_meas_id.tolist(),
                'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()],
            },
            observed_data={
                'observed_data': self.measurements.values
            },
            sample_stats={
                'lp': kernel_dist.squeeze(-1).T
            },
            posterior_predictive=sim_data,
            attrs=attrs
        )

    # @staticmethod
    def run_parallel(
            self,
            num_processes = 4,
            initial_points = None,
            n: int = 2000,
            n_burn = 0,
            thinning_factor=3,
            n_chains: int = 7,
            kernel=None,
            n_cdf=5,
            line_how='uniform',
            line_proposal_std=2.0,
            xch_how='gauss',
            xch_proposal_std=0.4,
            return_post_pred=False,
            evaluate_prior=False,
            kernel_kwargs=None
    ) -> az.InferenceData:
        mcmc_kwargs = dict(
            initial_points=initial_points,
            n=n,
            n_burn=n_burn,
            thinning_factor=thinning_factor,
            n_chains=n_chains,
            kernel=kernel,
            n_cdf=n_cdf,
            line_how=line_how,
            line_proposal_std=line_proposal_std,
            xch_how=xch_how,
            xch_proposal_std=xch_proposal_std,
            return_post_pred=return_post_pred,
            evaluate_prior=evaluate_prior,
            kernel_kwargs=kernel_kwargs
        )
        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        elif num_processes == 0:
            return self.run(**mcmc_kwargs)

        if num_processes > 0:
            pool = mp.Pool(num_processes)
            res = pool.starmap(self.run, [tuple(mcmc_kwargs.values()) for i in range(num_processes)])
            pool.close()
            pool.join()
            return az.concat(res, dim='chain')


class SMC_ABC(MCMC):
    # https://www.annualreviews.org/doi/pdf/10.1146/annurev-ecolsys-102209-144621
    #
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _BasePrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        for labelling_id, obsmod in mdv_observation_models.items():
            if obsmod.transformation_id is None:
                raise ValueError(f'Observationmodel {obsmod} does not have a transformation and therefore '
                                 f'euclidian distance is not defined (data lies on simplices)')
        super(SMC_ABC, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)

    def _sum_square_diff(self, data):
        # data = n_chains x n_samples x n_obs x n_data
        if len(data.shape) < 4:
            data = self._la.unsqueeze(data, -2)  # adds the n_obs dimension for simulations with n_obs=0 in the previous step
        return self._la.sum((data - self._x_meas) ** 2, -1, keepdims=False) # sum over n_data

    def _euclidean(self, data):
        diff_2 = self._sum_square_diff()

    def _rmse(self, data):
        diff_2 = self._sum_square_diff(data)
        return self._la.sqrt(self._la.sum(diff_2, -1, keepdims=True) / diff_2.shape[-1]) # sum over n_obs

    def rmse(
            self,
            fluxes,
            n_obs: int = 5,
            return_posterior_predictive: bool = False,
    ):
        if self._x_meas is None:
            raise ValueError('set measurement')

        sim_data = self.simulate(fluxes, n_obs)  # batch x n_obs x n_data
        # line below computes the root-mean-squared-error across simulated data and across measurements
        # rmse = batch x n_obs x n_meas x n_data
        distance = self._euclidean(sim_data)
        rmse = 1 / math.sqrt(distance.shape[-1]) * self._la.sum(distance, -1, keepdims=True)

        if return_posterior_predictive:
            return rmse, sim_data
        return rmse


    def epsilon_from_prior(
            self,
            simulated_data,
            alpha=0.05, # means that 5% of data in simulated data has a low anough
    ):
        rmse = self._rmse(simulated_data)

        max = simulated_data.max(1)[0].max(1)[0]
        min = simulated_data.min(1)[0].min(1)[0]
        diff = max - min
        print(torch.round(diff, decimals=3))
        print(torch.round(max, decimals=3))
        print(torch.round(min, decimals=3))
        print(torch.round(self._x_meas, decimals=3))
        print()

        print(rmse.min())
        print(rmse.max())
        # print(rmse, rmse.shape)

    def run(
            self,
            n_smc_steps=10,
            n=1000,
    ):
        for i in range(n_smc_steps):
            if i == 0:
                theta = self._prior.sample((n, ))
            else:
                pass

        pass



import holoviews as hv
hv.extension('bokeh')
class PlotMonster(object):
    _ALLFONTSIZES = {
        'xlabel': 12,
        'ylabel': 12,
        'zlabel': 12,
        'labels': 12,
        'xticks': 10,
        'yticks': 10,
        'zticks': 10,
        'ticks': 10,
        'minor_xticks': 8,
        'minor_yticks': 8,
        'minor_ticks': 8,
        'title': 14,
        'legend': 12,
        'legend_title': 12,
    }
    _FONTSIZES = {
        'labels': 14,
        'ticks': 12,
        'minor_ticks': 8,
        'title': 16,
        'legend': 12,
        'legend_title': 14,
    }
    def __init__(
            self,
            polytope: LabellingPolytope,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None
    ):
        self._pol = polytope
        self._data = inference_data

        prior_color = '#2855de'
        post_color = '#e02450'
        self._colors = {
            'true': '#ff0000',
            'map': '#13f269',
            'prior': prior_color,
            'prior_predictive': prior_color,
            'posterior': post_color,
            'posterior_predictive': post_color,
        }

        if not all(polytope.A.columns.isin(inference_data.posterior.theta_id.values)):
            raise ValueError

        if v_rep is None:
            v_rep = V_representation(polytope, number_type='fraction')
        else:
            if not v_rep.columns.equals(polytope.A.columns):
                raise ValueError

        self._v_rep = v_rep
        self._fva = fast_FVA(polytope)
        self._odf = self._load_observed_data()
        self._map = self._load_MAP()
        self._ttdf = self._load_true_theta()

    @property
    def obsdat_df(self):
        return self._odf

    def _axes_range(
            self,
            var_id,
            return_dimension=True,
            label=None,
            tol=12,
    ):
        fva_min = self._fva.loc[var_id, 'min']
        fva_max = self._fva.loc[var_id, 'max']

        if tol > 0:
            tol = abs(fva_min - fva_max) / 10

        range = (fva_min - tol, fva_max + tol)
        if return_dimension:
            kwargs = dict(spec=var_id, range=range)
            if label is not None:
                kwargs['label'] = label
            return hv.Dimension(**kwargs)
        return range

    def _process_points(self, points: np.ndarray):
        hull = ConvexHull(points)
        verts = hull.vertices.copy()
        verts = np.concatenate([verts, [verts[0]]])
        return hull.points[verts]

    def _get_samples(self, group='posterior', num_samples=None, *args):
        group_var_map = {
            'posterior': 'theta',
            'prior': 'theta',
            'posterior_predictive': 'simulated_data',
            'prior_predictive': 'simulated_data',
        }
        return az.extract(
            self._data,
            group=group,
            var_names=group_var_map[group],
            combined=True,
            num_samples=num_samples,
            rng=True,
        ).loc[list(args)].values.T

    def density_plot(
            self,
            var_id,
            num_samples=30000,
            group='posterior',
            bw=None,
            include_fva = True,
    ):
        sampled_points = self._get_samples(group, num_samples, var_id)
        if group in ['posterior', 'prior']:
            xax = self._axes_range(var_id)
        else:
            xax = hv.Dimension(var_id)
        plots = [
            hv.Distribution(sampled_points, kdims=[xax], label=group).opts(bandwidth=bw, color=self._colors[group])
        ]
        if include_fva and (group in ['posterior', 'prior']):
            fva_min, fva_max = self._axes_range(var_id, return_dimension=False, tol=0)
            opts = dict(color='#000000', line_dash='dashed')
            plots.extend([
                hv.VLine(fva_min).opts(**opts), hv.VLine(fva_max).opts(**opts),
            ])
        return hv.Overlay(plots).opts(xrotation=90, height=400, width=400, show_grid=True, fontsize=self._FONTSIZES)

    def _plot_area(self, vertices: np.ndarray, var1_id, var2_id, label=None, color='#ebb821'):
        xax = self._axes_range(var1_id)
        yax = self._axes_range(var2_id)
        plots = [
            hv.Area(vertices, kdims=[xax], vdims=[yax], label=label).opts(
                alpha=0.2, show_grid=True, width=800, height=600, color=color
            ),
            hv.Curve(vertices, kdims=[xax], vdims=[yax]).opts(color=color)
        ]
        return hv.Overlay(plots).opts(fontsize=self._FONTSIZES)

    def _plot_polytope_area(self, var1_id, var2_id):
        pol_verts = self._v_rep.loc[:, [var1_id, var2_id]].drop_duplicates()
        vertices = self._process_points(pol_verts.values)
        return self._plot_area(vertices, var1_id, var2_id, label='polytope')

    def _data_hull(
            self,
            var1_id,
            var2_id,
            group='posterior',
            num_samples=None
    ):
        sampled_points = self._get_samples(group, num_samples, var1_id, var2_id)
        vertices = self._process_points(sampled_points)
        return self._plot_area(vertices, var1_id, var2_id, label=f'{group} sampled support', color=self._colors[group])

    def _bivariate_plot(
            self,
            var1_id,
            var2_id,
            group='posterior',
            num_samples=30000,
            bandwidth=None,
    ):
        sampled_points = self._get_samples(group, num_samples, var1_id, var2_id)
        xax = self._axes_range(var1_id)
        yax = self._axes_range(var2_id)
        return hv.Bivariate(sampled_points, kdims=[xax, yax], label='density').opts(
            bandwidth=bandwidth, filled=True, alpha=1.0, cmap='Blues', fontsize=self._FONTSIZES
        )

    def _load_observed_data(self):
        measurement_id = self._data.observed_data['measurement_id']
        data_id = self._data.observed_data['data_id'].values
        return pd.DataFrame(
            self._data.observed_data['observed_data'].values, index=measurement_id, columns=data_id
        )

    def _load_MAP(self):
        lp = self._data.sample_stats.lp.values
        chain_idx, draw_idx = np.argwhere(lp == lp.max()).T
        row, col = chain_idx[0], draw_idx[0]
        max_lp = lp[row, col]

        theta_id = self._data['posterior']['theta_id'].values
        theta = pd.DataFrame(
            self._data['posterior']['theta'].values[row, col, :], index=theta_id
        ).T

        result = {'lp': max_lp, 'theta': theta}

        if 'posterior_predictive' in self._data:
            data_id = self._data['posterior_predictive']['data_id'].values
            data = pd.DataFrame(
                self._data['posterior_predictive']['simulated_data'].values[row, col, :], index=data_id
            ).T
            result['data']=data
        return result

    def _load_true_theta(self):
        theta_id = self._data.posterior['theta_id'].values
        true_theta = post.attrs.get('true_theta')
        if true_theta is None:
            return
        return pd.DataFrame(true_theta, index=theta_id).T

    def point_plot(self, var1_id, var2_id=None, what_var='theta', what_point='true'):
        if what_var == 'theta':
            xax = self._axes_range(var1_id)
            if var2_id is not None:
                yax = self._axes_range(var2_id)
        elif what_var == 'data':
            xax = hv.Dimension(var1_id)
            yax = hv.Dimension(var1_id)
        else:
            raise ValueError

        if what_point == 'map':
            if what_var not in self._map:
                raise ValueError(f'{what_var} not in InferenceData')
            to_plot = self._map[what_var]
        elif what_point == 'true':
            if self._ttdf is None:
                raise ValueError('no true theta in this InferenceData')
            if what_var == 'theta':
                to_plot = self._ttdf
            else:
                to_plot = self._odf
        if var2_id is None:
            return hv.VLine(to_plot.loc[:, var1_id].values).opts(
                color=self._colors[what_point], line_dash='dashed', xrotation=90
            )
        return hv.Points(to_plot.loc[:, [var1_id, var2_id]], kdims=[xax, yax], label=what_point).opts(
            color=self._colors[what_point], size=7, fontsize=self._FONTSIZES
        )

    # def observed_data_plot(self, var1_id, var2_id=None, what='map'):
    #     if var2_id is None:
    #         return hv.VLine(self.obsdat_df.loc[:, var1_id].values).opts(
    #             color=self._colors['true_theta'], line_dash='dashed', xrotation=90
    #         )
    #     return hv.Points(self.obsdat_df.loc[:, [var1_id, var2_id]], kdims=[var1_id, var2_id]).opts(
    #         color=self._colors['true_theta'], size=7, fontsize=self._FONTSIZES
    #     )

    def grand_data_plot(self, var_names: Iterable):
        plots = []
        cols = 3
        for i, var_id in enumerate(var_names):
            show_legend = True if i == cols - 1 else False
            postpred = pm.density_plot(var_id, group='posterior_predictive')
            priopred = pm.density_plot(var_id, group='prior_predictive')
            true = pm.point_plot(var_id, what_var='data', what_point='true')
            map = pm.point_plot(var_id, what_var='data', what_point='map')
            width = 600 if i % cols == cols - 1 else 400
            panel = (postpred * priopred * true * map).opts(
                legend_position='right', show_legend=show_legend, width=width, show_grid=True, fontsize=self._FONTSIZES,
                ylabel='',
            )
            plots.append(panel)

        return hv.Layout(plots).cols(cols)

    def grand_theta_plot(self, var1_id, var2_id, group='posterior'):
        plots = [
            pm._plot_polytope_area(var1_id, var2_id),
            pm._data_hull(var1_id=var1_id, var2_id=var2_id, group=group),
            pm._bivariate_plot(var1_id=var1_id, var2_id=var2_id, group=group),
        ]
        if group == 'posterior':
            plots.extend([
                pm.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='map'),
                pm.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='true')
            ])
        return hv.Overlay(plots).opts(legend_position='right', show_legend=True, fontsize=self._FONTSIZES)


if __name__ == "__main__":
    # from pta.sampling.uniform import sample_flux_space_uniform, UniformSamplingModel
    # from pta.sampling.tfs import TFSModel
    import pickle, os
    from sbmfi.models.small_models import spiro, multi_modal
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.core.util import _excel_polytope
    from sbmfi.core.observation import MVN_BoundaryObservationModel
    from sbmfi.settings import SIM_DIR
    from sbmfi.core.polytopia import FluxCoordinateMapper, compute_volume
    # from sbmfi.inference.simulator import MCMC
    from cdd import Fraction
    from bokeh.plotting import show, output_file
    from arviz import plot_density
    from holoviews.operation.stats import univariate_kde
    # hv.extension('bokeh')

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    model, kwargs = spiro(
        backend='torch',
        batch_size=1, which_measurements='lcms', build_simulator=True, which_labellings=list('CD'),
        v2_reversible=True, logit_xch_fluxes=False, include_bom=True, seed=None
    )
    sdf = kwargs['substrate_df']
    datasetsim = kwargs['datasetsim']
    simm = datasetsim._obmods
    bom = datasetsim._bom
    up = UniFluxPrior(model, cache_size=2000)
    # from botorch.utils.sampling import sample_polytope
    mcmc = SMC_ABC(
        model=model,
        substrate_df=sdf,
        mdv_observation_models=simm,
        boundary_observation_model=bom,
        prior=up,
    )
    algo = 'rej'
    mcmc.set_measurement(x_meas=kwargs['measurements'])
    mcmc.set_true_theta(theta=kwargs['theta'])
    run_kwargs = dict(
        epsilon=5.0,
        n=50, n_burn=0, n_chains=4, thinning_factor=2, n_cdf=4, return_post_pred=True, line_proposal_std=5.0,
        evaluate_prior=False, algorithm=algo
    )

    result = simulate_prior_predictive(mcmc, num_processes=0, n_obs=5, n=100, include_prior_predictive=True)
    mcmc.epsilon_from_prior(result['simulated_data'])

    # post = mcmc.run(**run_kwargs)
    # az.to_netcdf(post, filename=f'spiro_{algo}.nc')

    plot = False
    if plot:
        pickle.dump(model._fcm._sampler.basis_polytope, open('pol.p', 'wb'))
        pol = pickle.load(open('pol.p', 'rb'))
        # nc_file = "C:\python_projects\sbmfi\src\sbmfi\inference\e_coli_anton_glc7_prior.nc"
        nc_file = "C:\python_projects\sbmfi\spiro_cdf.nc"
        post = az.from_netcdf(nc_file)

        v_rep = None
        # v_rep = pd.read_excel('v_rep.xlsx', index_col=0)
        pm = PlotMonster(pol, post, v_rep=v_rep)
        pm._v_rep.to_excel('v_rep.xlsx')

        var1_id = 'B_svd2'
        var2_id = 'B_svd3'
        group = 'posterior'

        map = pm._load_MAP()
        measurements = pm._load_observed_data()
        boli = measurements.columns.str.contains('[CD]\+', regex=True)
        plot = pm.grand_data_plot(measurements.columns[boli])
        # hv.save(plot, 'pltts.png')

        # plot = pm.grand_theta_plot(var1_id, var2_id, group='prior')


        # aa = pm.plot_density('D: C+0', group='posterior_predictive', var_names='simulated_data')
        #
        # a = pm._plot_polytope_area(var1_id, var2_id)
        # b = pm._data_hull(var1_id=var1_id, var2_id=var2_id, group=group)
        # c = pm._bivariate_plot(var1_id=var1_id, var2_id=var2_id, group=group)
        # plot = a * b * c
        # if group == 'posterior':
        #     d = pm.point_plot(var1_id=var1_id, var2_id=var2_id, what='map')
        #     e = pm.point_plot(var1_id=var1_id, var2_id=var2_id, what='true_theta')
        #     plot = plot * d * e
        # plot = plot.opts(legend_position='right', show_legend=True)
        # d = pm.density_plot(var1_id)
        # e = pm.density_plot(var1_id, group=group)
        output_file('test.html')
        show(hv.render(plot))
        # show(hv.render(d))
