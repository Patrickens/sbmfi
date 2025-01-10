from sbmfi.priors.uniform import *

class ProjectionPrior(UniformRoundedFleXchPrior):
    # TODO I noticed that for the biomass flux, it is rarely sampled over 0.3, thus here we
    #  sample boundary fluxes in a projected polytope and then constrain and sample just like with ratios

    # PROBLEM if we sample uniformely in xch space, the resulting polytopes have different volumes
    #   which implies a different log_prob, we could compute the vol
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            projected_fluxes: Iterable,
            xch_prior: _BaseXchFluxPrior = None,
            projection_pol: LabellingPolytope = None,
            num_processes: int = 0,
            number_type='float',
    ):
        super(ProjectionPrior, self).__init__(model, xch_prior, num_processes=num_processes)
        raise NotImplementedError('does now work with new API')
        if projection_pol is None:
            pol = self._fcm._Fn
            settings = self._fcm._sampler._pr_settings
            spol = PolyRoundApi.simplify_polytope(pol, settings=settings, normalize=False)
            pol = LabellingPolytope.from_Polytope(spol, pol)
            P = pd.DataFrame(0.0, index=projected_fluxes, columns=pol.A.columns)
            P.loc[projected_fluxes, projected_fluxes] = np.eye(len(projected_fluxes))
            try:
                self._projection_pol = rref_and_project(pol, P=P, number_type=number_type, settings=settings)
            except:
                self._projection_pol = project_polytope(pol, P=P, number_type=number_type)
        else:
            if not projection_pol.A.columns.isin(projected_fluxes).all():
                raise ValueError(f'wrong projection pol: {projection_pol.A.columns}, '
                                 f'wrt projected fluxes {projected_fluxes}')
            self._projection_pol = projection_pol

        self._volumes = None
        self._boundary_psm = PolytopeSamplingModel(self._projection_pol)
        self._projection_initial_points = None
        self._projection_fva = fast_FVA(self._projection_pol)

    def _run_tasks(
            self, tasks, fn=sample_polytope, break_i=-1, scramble=True,
            what='rounded', return_results=False, n_tasks=0, desc=None
    ):
        if n_tasks:
            pbar = tqdm.tqdm(tasks, total=n_tasks, ncols=100, desc=desc)
        else:
            pbar = tasks

        if self._num_processes > 0:
            results = self._mp_pool.starmap(fn, pbar)
        else:
            results = []
            for i, task in enumerate(pbar):
                results.append(fn(*task))
                if (break_i > -1) and (i > break_i):
                    break
        if n_tasks:
            pbar.close()

        if fn == sample_polytope:
            whatensor = self._la.cat([torch.as_tensor(r[what]) for r in results])
            if scramble:
                scramble_indices = self._la.randperm(whatensor.shape[0])
                whatensor = whatensor[scramble_indices]
            if 'new_initial_points' in results[0]:
                self._initial_points = results[0]['new_initial_points']

            if return_results:
                log_det_E = [r['log_det_E'] for r in results]
                if 'psm' in results[0]:
                    psms = [r['psm'] for r in results]
                return whatensor, {'log_det_E': log_det_E, 'psms': psms}
            return whatensor

        elif fn == compute_volume:
            return pd.DataFrame(results)
        else:
            return results

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        raise NotImplementedError

    def _fill_caches(
            self, n=100, n_flux=10, rel_tol=0.01, break_i=-1, close_pool=True, scramble=False,
            what_volume='polytope', enumerate_vertices=False,
    ):
        boundary_result = sample_polytope(
            self._boundary_psm, n=n, initial_points=self._projection_initial_points, new_initial_points=True,
            return_what='basis',
        )
        self._projection_initial_points = boundary_result['new_initial_points']
        boundary_samples = self._boundary_psm.map_rounded_2_fluxes(boundary_result['basis'], pandalize=True)
        lb = boundary_samples.copy()
        ub = boundary_samples.copy()
        lb.columns += '|lb'
        ub.columns += '|ub'

        if rel_tol > 0.0:
            bounds_tol = (self._projection_fva['max'] - self._projection_fva['min']) * rel_tol * 0.5
            lb -= bounds_tol.values
            ub += bounds_tol.values
            lb = lb.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)
            ub = ub.clip(lower=self._projection_fva['min'].values, upper=self._projection_fva['max'].values, axis=1)

        b_constraint_df = pd.concat([-lb, ub], axis=1)  # NB this dataframe contains all the bounds for the b vector

        kernel_id = 'rref' if enumerate_vertices else 'svd'
        coordinate_id = 'transformed' if enumerate_vertices else 'rounded'
        sampling_task_generator = sampling_tasks(
            self._fcm._Fn, b_constraint_df=b_constraint_df, counts=n_flux, return_what='net_fluxes',
            return_psm=True, kernel_id=kernel_id, coordinate_id=coordinate_id,
        )

        net_fluxes = self._run_tasks(
            sampling_task_generator, break_i=break_i, close_pool=close_pool and what_volume is None, scramble=True,
            what='net_fluxes', return_results=True, n_tasks=n, desc='sampling fluxes'
        )

        net_fluxes, results = net_fluxes
        if what_volume is None:
            pass
        else:
            if what_volume == 'polytope':
                volume_task_generator = volume_tasks(results['psms'], enumerate_vertices=enumerate_vertices)
                volume_df = self._run_tasks(
                    volume_task_generator, fn=compute_volume, break_i=break_i, close_pool=close_pool,
                    n_tasks=n, desc='computing volumes'
                )
                volume_df = pd.concat([boundary_samples, volume_df], axis=1)
            elif what_volume == 'log_det_E':
                boundary_samples['log_det_E'] = results['log_det_E']
                volume_df = boundary_samples

            if self._volumes is None:
                self._volumes = volume_df
            else:
                self._volumes = pd.concat([self._volumes, volume_df], axis=0, ignore_index=True)

        theta = self._fcm._sampler.map_fluxes_2_rounded(net_fluxes)
        if self._fcm._nx > 0:
            xch_basis_samples = self._xch_prior.sample((n * n_flux, ))
            theta = self._la.cat([theta, xch_basis_samples], dim=-1)

        self._theta_cache = theta


if __name__ == "__main__":
    pass