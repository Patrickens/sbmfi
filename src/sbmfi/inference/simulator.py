import tables as pt
import warnings
import torch
import psutil
import math
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Iterable, Union, Dict, Tuple
from collections import OrderedDict
import tqdm
from sbmfi.core.model import LabellingModel
from sbmfi.core.simulfuncs import (
    init_observer,
    obervervator_worker,
    observator_tasks,
)
from sbmfi.core.observation import (
    BoundaryObservationModel,
    MDV_ObservationModel,
)
from sbmfi.core.util import (
    _excel_polytope,
    hdf_opener_and_closer,
    _bigg_compartment_ids,
    make_multidex,
    profile,
)


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

# from line_profiler import line_profiler
# import arviz as az
# import tqdm
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
            raise ValueError(f'non-unique identifiers for labelling! {substrate_df.index}')

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
        self._fcm = model._fcm
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
        return self._fcm.theta_id

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
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self._fcm.fluxes_id].values)

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

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            j, k = self._obsize[labelling_id]
            self._model.set_input_labelling(input_labelling=self._substrate_df.loc[labelling_id])
            mdv = self._model.cascade()
            if return_mdvs:
                result[:, i*self._model._ns : (i+1) * self._model._ns] = mdv
            else:
                result[:, slicer, j:k] = obmod(mdv, n_obs=n_obs)

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

    def to_partial_mdvs(self, data, is_mdv=False, normalize=False, pandalize=True):
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = self._la.get_tensor(values=data.values)

        processed = []
        columns = {}
        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            if is_mdv:
                processed.append(obmod.compute_observations(data[:, i, :]))
            else:
                if obmod._transformation is None:
                    raise ValueError(
                        'obmod does not have transformation specified, perhaps data is not log-ratio transformed'
                    )
                j, k = self._obsize[labelling_id]
                part_mdvs = obmod._transformation.inv(data[..., j:k])  # = intensities
                if normalize:
                    part_mdvs = obmod.compute_observations(part_mdvs)
                processed.append(part_mdvs)
            columns[labelling_id] = obmod._observation_df.index.copy()
        if self._bomsize > 0:
            processed.append(data[..., -self._bomsize:])
            columns['BOM'] = self._bom.boundary_id
        processed = self._la.cat(processed, -1)
        if pandalize:
            processed = pd.DataFrame(self._la.tonp(processed), index=index, columns=make_multidex(columns, name1='data_id'))
        return processed

    def prepare_for_sbi(
            self,
            theta,
            data,
            randomize=True,
            **kwargs # this allows to pass **result
    ):
        # cast to correct datatype and I guess do some other pre-processing?
        # TODO think of device!
        data = torch.as_tensor(data, dtype=torch.float32)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        n_t = theta.shape[-1]
        n_samples, n_obs, n_x = data.shape
        theta = theta.unsqueeze(1).repeat((1, n_obs, 1)).view(n_obs * n_samples, n_t)
        data = data.view(n_obs * n_samples, n_x)
        if randomize:
            ridx = torch.as_tensor(self._la.choice(n_samples, n_samples))
            data = data[ridx]
            theta = theta[ridx]
        return theta, data

    def _verify_hdf(self, hdf: pt.file):
        substrate_df = pd.read_hdf(hdf.filename, key='substrate_df', mode=hdf.mode)
        if not self._substrate_df.equals(substrate_df):
            raise ValueError('hdf has different substrate_df')
        for what, compare in {
            'fluxes': self._fcm.fluxes_id,
            'mdv': self._model.state_id,
            'data': self.data_id,
            'theta': self.theta_id,
        }.items():
            if what == 'data':
                what_id = pd.MultiIndex.from_frame(pd.read_hdf(hdf.filename, key='data_id', mode=hdf.mode))
            else:
                what_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
            if not compare.equals(what_id):
                raise ValueError(f'{what}_id is different between model and hdf')

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
            pt.Array(hdf.root, name='theta_id', obj=self.theta_id.values.astype(str))
            pt.Array(hdf.root, name='fluxes_id', obj=self._model._fcm.fluxes_id.values.astype(str))  # NB these are the untrimmed fluxes
            self.data_id.to_frame(index=False).to_hdf(hdf.filename, key='data_id', mode=hdf.mode, format='table')
        else:
            self._verify_hdf(hdf)

        if (dataset_id in hdf.root) and not append:
            hdf.remove_node(hdf.root, name=dataset_id, recursive=True)

        if dataset_id not in hdf.root:
            hdf.create_group(hdf.root, name=dataset_id)

        dataset_children = hdf.root[dataset_id]._v_children
        if (len(dataset_children) > 0) and (dataset_children.keys() != result.keys()):
            raise ValueError(f'result {result.keys()} has different data than dataset {dataset_children.keys()}; cannot append!')

        dataset_shapes = []
        for item, array in result.items():
            if isinstance(array, pd.DataFrame):
                array = array.values
            if not isinstance(array, np.ndarray):
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
            dataset_shapes.append(ptarray.shape[0])
        if not all(np.array(dataset_shapes) == dataset_shapes[0]):
            raise ValueError(f'unbalanced dataset: {dataset_shapes}')

    @hdf_opener_and_closer(mode='r')
    def read_hdf(
            self,
            hdf:   str,
            dataset_id: str,
            what:  str,
            labelling_id: Union[str, Iterable[str]] = None,
            start: int = None,
            stop:  int = None,
            step:  int = None,
            pandalize: bool = False,
    ) -> Union[np.array, pd.DataFrame]:
        if (dataset_id not in hdf.root):
            raise ValueError(f'{dataset_id} not in hdf')
        elif (what not in hdf.root[dataset_id]):
            raise ValueError(f'{what} not in {dataset_id}')

        self._verify_hdf(hdf)

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

        if what in ('fluxes', 'theta'):
            xcs_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
            return pd.DataFrame(xcsarr, index=samples_id, columns=xcs_id), validx

        elif what == 'validx':
            return validx, validx

        elif what == 'data':
            i_obs = (*range(xcsarr.shape[1]), )
            dataframes = [pd.DataFrame(xcsarr[:, i, :], index=samples_id, columns=self.data_id) for i in i_obs]
            dataframe = pd.concat(dataframes, keys=i_obs, names=['i_obs', 'samples_id'])
            return dataframe.swaplevel(0, 1, 0).sort_values(by=['samples_id', 'i_obs'], axis=0).loc[:, labelling_id], validx

        elif what == 'mdv':
            return pd.concat([
                pd.DataFrame(xcsarr[:, i], index=samples_id, columns=self._model.state_id)
                for i in labelling_idx], axis=1, keys=labelling_id
            ), validx

    def sbi_data_from_hdf(
            self,
            hdf: str,
            dataset_id: str,
            n: int = None,
            device=None,
    ):
        data, validx = self.read_hdf(hdf, dataset_id, what='data')
        data = data[validx.all(-1)]
        theta, validx = self.read_hdf(hdf, dataset_id, what='theta')
        theta = theta[validx.all(-1)]

        if n is not None:
            if n > data.shape[0]:
                raise ValueError(f'cannot sample {n} from dataset of lenghth {data.shape[0]}')
            ridx = self._la.tonp(self._la.choice(n, data.shape[0]))
            data = data[ridx]
            theta = theta[ridx]
        return self.prepare_for_sbi(theta, data, device)

    def __call__(self, theta, n_obs=3, **kwargs):
        fluxes = self._fcm.map_theta_2_fluxes(theta)
        return self.simulate(fluxes, n_obs, **kwargs)


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
            elif (key == 'validx'):
                continue
            else:
                chunk = worker_result.get(f'{key}_chunk')
                if chunk is None:
                    continue
                result[key][start_stop_idx, i_obs] = chunk

    def simulate_set(
            self,
            fluxes,
            n_obs=3,
            fluxes_per_task=None,
            what='data',
            break_i=-1,
            close_pool=True,
            show_progress=False,
    ) -> OrderedDict:
        # TODO perhaps also parse samples_id
        result = {}
        result['validx'] = self._la.get_tensor(shape=(fluxes.shape[0], len(self._obmods)), dtype=np.bool_)
        result['fluxes'] = fluxes # save before trimming

        fluxes = self._model._fcm.frame_fluxes(fluxes, trim=True)

        if what not in ('all', 'data', 'mdv'):
            raise ValueError('not sure what to simulate')
        if fluxes.shape[0] < self._la._batch_size:
            raise ValueError(f'n must be at least batch size: {self._la._batch_size}')

        if what != 'data':
            result['mdv'] = self._la.get_tensor(shape=(fluxes.shape[0], len(self._obmods), len(self._model.state_id)))
        if what != 'mdv':
            n_obshape = max(1, n_obs)
            result['data'] = self._la.get_tensor(shape=(fluxes.shape[0], n_obshape, len(self.data_id)))

        if (self._bomsize > 0) and (what != 'mdv'):
            slicer = 0 if n_obs == 0 else slice(None)
            bo_fluxes = fluxes[:, self._bo_idx]  # TODO only works with torch now, since we sample from a torch distribution!
            result['data'][:, slicer, -self._bomsize:] = self._bom(bo_fluxes, n_obs=n_obs)

        if fluxes_per_task is None:
            fluxes_per_task = math.ceil(fluxes.shape[0] / max(self._num_processes, 1))

        fluxes_per_task = min(fluxes.shape[0], fluxes_per_task)

        tasks = observator_tasks(
            fluxes, substrate_df=self._substrate_df, fluxes_per_task=fluxes_per_task, n_obs=n_obs, what=what
        )
        if show_progress:
            pbar = tqdm.tqdm(total=fluxes.shape[0], ncols=100)

        if self._num_processes == 0:
            init_observer(self._model, self._obmods, self._eps)
            for i, task in enumerate(tasks):
                worker_result = obervervator_worker(task)
                self._fill_results(result, worker_result)
                if show_progress:
                    i, j = worker_result['start_stop']
                    pbar.update(n = j - i)
                # self._fill_results(result, obervervator_worker(task))
                if (break_i > -1) and (i > break_i):
                    break
        else:
            mp_pool = self._get_mp_pool()
            for worker_result in mp_pool.imap_unordered(obervervator_worker, iterable=tasks):
                self._fill_results(result, worker_result)
                if show_progress:
                    i, j = worker_result['start_stop']
                    pbar.update(n = j - i)
            if close_pool:
                mp_pool.close()
                mp_pool.join()
        if show_progress:
            pbar.close()
        return result

    def __call__(self, theta, n_obs=5, fluxes_per_task=None, close_pool=False, show_progress=False, **kwargs):
        fluxes = self._model._fcm.map_theta_2_fluxes(theta)
        result = self.simulate_set(
            fluxes, n_obs,
            fluxes_per_task=fluxes_per_task,
            what='data',
            close_pool=close_pool,
            show_progress=show_progress,
            **kwargs
        )
        return result['data']


if __name__ == "__main__":
    pass
