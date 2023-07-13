from multiprocessing import Pool
# from torch.multiprocessing import Pool  # TODO: make this corresponding to the backend?
import tables as pt
import warnings
import torch
import time
import logging
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
from functools import lru_cache
from PolyRound.api import PolyRoundApi
# from torch.distributions import Distribution  # TODO perhaps make this the base class of _BaseSimulator
from sbmfi.core.model import EMU_Model, LabellingModel, RatioMixin
from sbmfi.core.simulfuncs import (
    init_observer,
    obervervator_worker,
    observator_tasks,
    designer_worker,
    designer_tasks,
)
from sbmfi.core.observation import (
    LCMS_ObservationModel,
    BoundaryObservationModel,
    exclude_low_massiso,
    ClassicalObservationModel,
    TOF6546Alaa5minParameters,
    MDV_ObservationModel,
    _BlockDiagGaussian,
    MDV_LogRatioTransform,

)
from pta.sampling.commons import (
    SamplingResult, split_chains, apply_to_chains, fill_common_sampling_settings, sample_from_chains
)
from sbmfi.estimate.priors import (
    _BasePrior,
    UniFluxPrior,
    ThermoPrior,
    RatioPrior,
    _CannonicalPolytopeSupport,
)
from pta.constants import (
    default_min_chains,
    us_steps_multiplier,
)
from sbmfi.core.util import (
    _excel_polytope,
    hdf_opener_and_closer,
    _bigg_compartment_ids,
    make_multidex,
    profile,
)
from line_profiler import line_profiler
from sbmfi.core.polytopia import PolytopeSamplingModel
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
"""

# prof2 = line_profiler.LineProfiler()
class _BaseSimulator(object):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        if not model._is_built:
            raise ValueError('need to build model')
        if not substrate_df.index.unique().all():
            raise ValueError('non-unique identifiers for labelling!')

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
        n_f = fluxes.shape[0]
        self._model.set_fluxes(fluxes, index, trim=True)

        if return_mdvs:
            n_state = len(self._model.state_id)
            result = self._la.get_tensor(shape=(n_f, n_state * len(self._obmods)))
        else:
            result = self._la.get_tensor(shape=(n_f, n_obshape, len(self.data_id)))
            if self._bomsize > 0:
                result[:, :, -self._bomsize:] = self._bom.sample_observation(
                    self._model._fluxes[:, self._bo_idx], n_obs=n_obs
                )

        i=0
        for j, (labelling_id, obmod) in enumerate(self._obmods.items()):
            self._model.set_input_labelling(input_labelling=self._substrate_df.loc[labelling_id])
            mdv = self._model.cascade()
            if return_mdvs:
                result[:, j*n_state : (j+1) * n_state] = mdv
            else:
                result[:, :, i:i+obmod._n_d] = obmod(mdv, n_obs=n_obs)
                i += obmod._n_d

        if pandalize:
            if return_mdvs:
                columns = make_multidex({k: self._model.state_id for k in self._obmods}, 'labelling_id', 'mdv_id')
                result = pd.DataFrame(result, index=index, columns=columns)
            else:
                result = self._la.tonp(result).transpose(1, 0, 2).reshape((n_f * n_obshape, len(self.data_id)))
                if index is None:
                    index = pd.RangeIndex(n_f)
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
        self._true_theta = self._la.atleast_2d(self._la.get_tensor(values=theta.loc[self._model._fcm.theta_id].values))

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
        return pd.Series(self._la.tonp(self._true_theta), index=self._model._fcm.theta_id, name='true_theta')


class DataSetSim(_BaseSimulator):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            boundary_observation_model: BoundaryObservationModel = None,
            num_processes=0,
            epsilon=1e-12,
    ):
        super(DataSetSim, self).__init__(model, substrate_df, mdv_observation_models, boundary_observation_model)
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
            what='all',
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
            result['data'] = self._la.get_tensor(shape=(fluxes.shape[0], n_obs, len(self.data_id)))

        if self._bomsize > 0:
            bo_fluxes = fluxes[:, self._bo_idx]  # TODO only works with torch now, since we sample from a torch distribution!
            result['data'][:, :, -self._bomsize:] = self._bom(bo_fluxes, n_obs=n_obs)

        if fluxes_per_task is None:
            fluxes_per_task = math.ceil(fluxes.shape[0] / max(self._num_processes, 1))

        tasks = observator_tasks(
            fluxes, substrate_df=self._substrate_df, fluxes_per_task=fluxes_per_task, n_obs=n_obs, what=what
        )

        if self._num_processes == 0:
            init_observer(self._model, self._obmods, self._eps)
            for i, task in enumerate(tasks):
                ress = obervervator_worker(task)
                self._fill_results(result, ress)
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


class MCMC(_BaseSimulator):
    # TODO think about implementing MALA, this only requires the computation of gradients of log_prob

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _BasePrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        super(MCMC, self).__init__(model, substrate_df, mdv_observation_models, boundary_observation_model)
        self._prior = prior
        if prior is not None:
            if not prior._fcm.labelling_fluxes_id.equals(model.fluxes_id):
                raise ValueError('prior has different labelling fluxes than model')
            if not model._fcm.theta_id.equals(prior.theta_id):
                raise ValueError('theta of model and prior are different')
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

        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self._model._fcm.fluxes_id].values)

        self._model.set_fluxes(fluxes, index)

        n_f = fluxes.shape[0]
        n_meas = self._x_meas.shape[0]
        n_bom = 1 if self._bomsize > 0 else 0

        log_lik = self._la.get_tensor(shape=(n_f, n_meas, len(self._obmods) + n_bom))
        if return_posterior_predictive:
            mu_o = self._la.get_tensor(shape=(n_f, len(self.data_id)))

        if self._bomsize > 0:
            bo_meas = self._x_meas[:, -self._bomsize:]
            mu_bo = self._model._fluxes[:, self._bo_idx]
            log_lik[..., -1] = self._bom.log_lik(bo_meas=bo_meas, mu_bo=mu_bo)
            if return_posterior_predictive:
                mu_o[..., -self._bomsize:] = mu_bo

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            self._model.set_input_labelling(input_labelling=self._substrate_df.loc[labelling_id])
            mdv = self._model.cascade()
            j, k = self._obsize[labelling_id]
            x_meas_o = self._x_meas[..., j:k]
            ll = obmod.log_lik(x_meas_o, mdv, return_posterior_predictive)

            if return_posterior_predictive:
                ll, mo = ll
                mu_o[..., j:k] = mo

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
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = value.shape
        viewlue = self._la.view(value, shape=(self._la.prod(vape[:-1]), vape[-1]))

        n_f = viewlue.shape[0]
        k = len(self._obmods) + (1 if self._bom is None else 2) # the 2 is for a column of prior and boundary probabilities
        n_meas = self._x_meas.shape[0]
        log_prob = self._la.get_tensor(shape=(n_f, n_meas, k))

        fluxes = self._fcm.map_theta_2_fluxes(viewlue)

        if self._prior is not None:
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
            x0 = None,
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
            kernel_kwargs=None
    ):
        # TODO: this publication talks about this algo, but has a different acceptance procedure:
        #  doi:10.1080/01621459.2000.10473908
        #  doi:10.1007/BF02591694  Rinooy Kan article

        batch_size = n_chains * n_cdf
        if self._la._batch_size != batch_size:
            # this way the batch processing is corrected
            self._la._batch_size = batch_size
            self._model.build_simulator(**self._fcm.fcm_kwargs)

        K = self._sampler.dimensionality

        chains = self._la.get_tensor(shape=(n, n_chains, len(self._fcm.theta_id)))
        kernel_dist = self._la.get_tensor(shape=(n, n_chains, 1))

        sim_data = None
        if return_post_pred:
            sim_data = self._la.get_tensor(shape=(n, n_chains, len(self.data_id)))

        if x0 is None:
            net_basis_points = self._sampler.get_initial_points(num_points=n_chains)
            x = self._fcm.append_xch_flux_samples(net_basis_samples=net_basis_points, return_type='theta')
        else:
            x = x0

        x = self._la.tile(x, (n_cdf, 1))  # remember that the new batch size is n_chains x n_cdf

        ker_kwargs = {'return_posterior_predictive': return_post_pred}
        if kernel is None:
            kernel = self.log_prob
            if kernel_kwargs is not None:
                ker_kwargs = {**ker_kwargs, **kernel_kwargs}

        line_xs = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self._fcm.theta_id)))
        line_kernel_dist = self._la.get_tensor(shape=(1 + n_cdf, n_chains, 1))
        dist = kernel(x, **ker_kwargs)
        if return_post_pred:
            line_data = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self.data_id)))
            dist, data = dist

        line_kernel_dist[0] = dist[:n_chains]  # ordering of the samples from the PDF does not matter for inverse sampling
        x = x[: n_chains, :]
        log_probs_selecta = self._la.arange(n_chains)

        n_rev = len(self._model._fcm._fwd_id)

        n_tot = n_burn + n * thinning_factor
        biatch = min(2500, n_tot)
        for i in tqdm.trange(n_tot):  # for i, (ar, r, rnd) in enumerate(zip(ARs, Rs, rands)) is reaaaally fucking slow
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

            pol_dist = self._sampler._h - self._sampler._G @ x[..., :K].T
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
            net_basis_points = x[..., :K] + line_alphas[..., None] * sphere_sample

            xch_fluxes = None
            if n_rev > 0:
                current_xch = x[..., -n_rev:]
                if self._fcm.logit_xch_fluxes:
                    current_xch = self._fcm._sigmoid_xch(current_xch)
                xch_fluxes = self._la.sample_bounded_distribution(
                    shape=(n_cdf, n_chains), lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                    mu=current_xch, which=xch_how, std=xch_proposal_std
                )

            line_xs[1:] = self._fcm.append_xch_flux_samples(
                net_basis_samples=net_basis_points, xch_fluxes=xch_fluxes, return_type='theta'
            )
            dist = self.log_prob(line_xs[1:], return_post_pred)
            if return_post_pred:
                dist, data = dist
                line_data[1:] = data

            line_kernel_dist[1:] = dist
            max_line_probs = line_kernel_dist.max(0)
            if isinstance(max_line_probs, tuple):
                max_line_probs = max_line_probs[0]

            normalized = line_kernel_dist - max_line_probs[None, :]
            probs = self._la.exp(normalized)  # TODO make sure this does not underflow!
            cdf = self._la.cumsum(probs, 0)  # empirical CDF
            cdf = cdf / cdf[-1, :]  # numbers between 0 and 1, now find the one closest to rnd to determine which sample is accepted

            # make sure that we select the 'next point' instead of the closest one!
            number_picker = cdf - rnd[None, :, None]
            number_picker[number_picker < 0.0] = float('inf')
            accept_idx = self._la.argmin(number_picker, 0, keepdims=False)  # indices of accepted samples
            accepted_probs = line_kernel_dist[accept_idx[:, 0], log_probs_selecta]
            line_kernel_dist[0] = accepted_probs # set the log-probs of the current sample
            x = line_xs[accept_idx[:, 0], log_probs_selecta]
            line_xs[0, :] = x  # set the log-probs of the current sample

            j = i - n_burn
            if j % thinning_factor == 0:
                k = j // thinning_factor
                kernel_dist[k] = accepted_probs
                chains[k] = x
                if return_post_pred:
                    sim_data[k] = line_data[accept_idx[:, 0], log_probs_selecta]

        if return_post_pred:
            sim_data = {
                'simulated_data': self._la.transax(sim_data, dim0=1, dim1=0)
            }
        return az.from_dict(
            posterior={
                'param': self._la.transax(chains, dim0=1, dim1=0)  # chains x draws x param
            },
            dims={
                'param': ['theta_id'],
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
        )



class MCMC_ABC(MCMC):
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
        super(MCMC_ABC, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)

    def euclidian(
            self,
            fluxes,
            n_obs=5,
            return_posterior_predictive=False,
    ):
        if self._x_meas is None:
            raise ValueError('set measurement')

        sim_data = self.simulate(fluxes, n_obs)  # batch x n_obs x n_data
        # line below computes the root-mean-squared-error across simulated data and across measurements
        rmse = self._la.sqrt(self._la.mean(((sim_data[:, :, None, :] - self._x_meas) **2), 1, keepdims=False))
        distance = self._la.sum(rmse, (1, 2), keepdims=False)

        if return_posterior_predictive:
            return distance, sim_data
        return distance

    def log_prob(
            self,
            value,
            return_posterior_predictive=False,
    ):
        pass





if __name__ == "__main__":
    from pta.sampling.uniform import sample_flux_space_uniform, UniformSamplingModel
    from pta.sampling.tfs import TFSModel
    import pickle, os
    from sbmfi.models.small_models import spiro, multi_modal
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.core.util import _excel_polytope
    from sbmfi.core.observation import MVN_BoundaryObservationModel
    from sbmfi.settings import SIM_DIR
    from sbmfi.core.polytopia import FluxCoordinateMapper, compute_volume



    bs = 10
    model, kwargs = spiro(batch_size=3, which_measurements='com', build_simulator=True,
                          which_labellings=list('CD'), v2_reversible=True, logit_xch_fluxes=True, include_bom=False)
    # model, kwargs = build_e_coli_anton_glc(backend='numpy', build_simulator=True, batch_size=bs, which_measurements='anton')

    sdf = kwargs['substrate_df']
    simm = kwargs['datasetsim']._obmods
    bom = kwargs['datasetsim']._bom

    mcmc = MCMC(
        model=model,
        substrate_df=sdf,
        mdv_observation_models=simm,
        boundary_observation_model=bom
    )
    mcmc.set_measurement(x_meas=kwargs['measurements'])
    post = mcmc.run(n=100, n_chains=3, thinning_factor=3, n_cdf=10, return_post_pred=True, line_proposal_std=5.0)

    # model, kwargs = spiro(batch_size=3, which_measurements='lcms', build_simulator=True,
    #                       which_labellings=list('CD'), v2_reversible=True, logit_xch_fluxes=True)
    # sdf = kwargs['substrate_df']
    # simm = kwargs['datasetsim']._obmods
    # bom = kwargs['datasetsim']._bom
    #
    # abc = MCMC_ABC(
    #     model=model,
    #     substrate_df=sdf,
    #     mdv_observation_models=simm,
    #     boundary_observation_model=bom
    # )
    # measurements = kwargs['measurements'].iloc[[0]]
    # abc.set_measurement(x_meas=measurements)
    #
    # fluxes = kwargs['fluxes'].loc[model._fcm.fluxes_id].to_frame().values
    # # fluxes = model._la.get_tensor(values=fluxes.values)[None, :]
    # fluxes = abc._la.tile(fluxes, (abc._la._batch_size, )).T
    # abc.euclidian(fluxes, n_obs=5)



    # az.plot_trace(
    #     post,
    # )



    # t, f = prior.sample_dataframes(5)
    # f.to_excel('f2.xlsx')
    # dss = DataSetSim(model, sdf, simm, bom, num_processes=2)
    # f = pd.read_excel('f2.xlsx', index_col=0)

    # co2 = model.metabolites.get_by_id('co2_c')
    # co2_consume = []
    # co2_produce = []
    # for reaction in co2._reaction:
    #     if reaction.id not in f.columns:
    #         continue
    #     if co2 in reaction.reactants:
    #         co2_consume.append(reaction.id)
    #     elif co2 in reaction.products:
    #         co2_produce.append(reaction.id)
    #
    # f.loc[:, co2_consume].to_excel('co2_consume.xlsx')
    # f.loc[:, co2_produce].to_excel('co2_produce.xlsx')
    # print(f.loc[:, co2_consume])
    # print(f.loc[:, co2_consume].sum(1))
    # print(f.loc[:, co2_produce])
    # print(f.loc[:, co2_produce].sum(1))


    # result = dss.simulate_set(fluxes=f, n_obs=3)
    # model.pretty_cascade(1)['A'].loc[(0, slice(None))].to_excel('a_1_1.xlsx')
    # model.reactions.get_by_id('GLYCL').pretty_tensors(1)['A'].to_excel('glycl_a_2.xlsx')

    # result['fluxes'] = f.values
    #
    # file = 'test_tomek.h5'
    # if os.path.exists(file):
    #     os.remove(file)
    #
    # dss.to_hdf(file, result, 'ding')
    # a, v = DataSetSim.read_hdf('test_tomek.h5', 'ding', 'mdv', pandalize=True)
    # print(1 - (model._sum @ a.T))
    # aa = DataSetSim.read_hdf('test.h5', 'ding', 'validx', pandalize=True)


