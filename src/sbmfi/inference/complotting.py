from sbmfi.core.polytopia import PolytopeSamplingModel, V_representation, fast_FVA, LabellingPolytope
from sbmfi.inference.arviz_monkey import *
import pandas as pd
import holoviews as hv
from scipy.spatial import ConvexHull
import numpy as np
from typing import Iterable, Union, Dict, Tuple
from bokeh.plotting import show
from bokeh.io import output_file
import matplotlib.pyplot as plt
import colorcet
from sbmfi.core.polytopia import thermo_2_net_polytope

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
            v_rep: pd.DataFrame = None,
            hv_backend='matplotlib',
    ):
        # TODO make sure we can plot exchange fluxes!
        hv.extension(hv_backend)
        self._hvb = hv_backend == 'matplotlib'
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
        self._group_var_map = {
            'posterior': 'theta',
            'prior': 'theta',
            'posterior_predictive': 'data',
            'prior_predictive': 'data',
        }
        self._px = 1/plt.rcParams['figure.dpi']

        if not all(polytope.A.columns.isin(inference_data.posterior.theta_id.values)):
            raise ValueError

        net_pol = thermo_2_net_polytope(polytope)
        if v_rep is None:
            v_rep = V_representation(polytope, number_type='fraction')
        else:
            if not v_rep.columns.equals(polytope.A.columns):
                raise ValueError

        self._v_rep = v_rep
        self._fva = fast_FVA(polytope)
        self._odf = self._load_observed_data()
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

    def _get_samples(self, *args, group='posterior', num_samples=None):
        print(*args)
        return az.extract(
            self._data,
            group=group,
            var_names=self._group_var_map[group],
            combined=True,
            num_samples=num_samples,
            rng=True,
        ).loc[list(args)].values.T

    def _plot_distribution(
            self,
            to_plot: pd.DataFrame,
            var_id,
            label=None,
            show_grid=True,
            bandwidth=None,
            color=None,
            muted=False,
            alpha=0.3,
            muted_alpha=0.07,
            show_legend=True,
            **opts
    ):
        kwargs = locals()
        kwargs.pop('opts')
        kwargs = {k: v for i, (k, v) in enumerate(kwargs.items()) if i > 3}
        kwargs = {**kwargs, **self._size_opts(), **opts}
        if self._hvb:
            kwargs.pop('muted', None)
            kwargs.pop('muted_alpha', None)
        return hv.Distribution(to_plot, kdims=[var_id], label=label).opts(**kwargs)

    def _get_chain(self, *args, chain_idx=0, group='posterior'):
        var = self._group_var_map[group]
        return self._data[group][var].sel({f'{var}_id': list(args)}).values[chain_idx]

    def _size_opts(self, width=500, height=400):
        kwargs = dict(height=height, width=width,)
        if self._hvb:
            kwargs = dict(fig_inches=(height * self._px, width * self._px),)
        return kwargs

    def density_plot(
            self,
            var_id,
            num_samples=30000,
            group='posterior',
            include_fva = True,
    ):
        sampled_points = self._get_samples(var_id, group=group, num_samples=num_samples)
        if group in ['posterior', 'prior']:
            xax = self._axes_range(var_id)
        else:
            xax = hv.Dimension(var_id)
        plots = [
            self._plot_distribution(sampled_points, xax, group, color=self._colors[group])
        ]
        if include_fva and (group in ['posterior', 'prior']):
            fva_min, fva_max = self._axes_range(var_id, return_dimension=False, tol=0)
            opts = dict(color='#000000', line_dash='dashed')
            if self._hvb:
                opts['linestyle'] = opts.pop('line_dash')
            plots.extend([
                hv.VLine(fva_min).opts(**opts), hv.VLine(fva_max).opts(**opts),
            ])

        return hv.Overlay(plots).opts(xrotation=90, show_grid=True, fontsize=self._FONTSIZES, show_legend=True, **self._size_opts())

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
        sampled_points = self._get_samples(var1_id, var2_id, group=group, num_samples=num_samples)
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
        sampled_points = self._get_samples(var1_id, var2_id, group=group, num_samples=num_samples)
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

    def _load_true_theta(self):
        theta_id = self._data.posterior['theta_id'].values
        true_theta = self._data.attrs.get('true_theta')
        if true_theta is None:
            return
        return pd.DataFrame(true_theta, index=theta_id).T

    def point_plot(self, var1_id, var2_id=None, what_var='theta', what_point='true', label=None):
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

        if label is None:
            label = what_point
        if var2_id is None:
            data = to_plot.loc[:, var1_id].values
            opts = dict(color=self._colors[what_point], line_dash='dashed', xrotation=90, line_width=2.5, show_legend=True)
            if self._hvb:
                opts['linewidth'] = opts.pop('line_width')
                opts['linestyle'] = opts.pop('line_dash')
            return hv.VLine(data).opts(**opts) * hv.Spikes(data, label=label).opts(**opts)
        return hv.Points(to_plot.loc[:, [var1_id, var2_id]], kdims=[xax, yax], label=what_point).opts(
            color=self._colors[what_point], size=7, fontsize=self._FONTSIZES
        )

    def observed_data_plot(self, var1_id, var2_id=None, what='map'):
        if var2_id is None:
            return hv.VLine(self.obsdat_df.loc[:, var1_id].values).opts(
                color=self._colors['true_theta'], line_dash='dashed', xrotation=90
            )
        return hv.Points(self.obsdat_df.loc[:, [var1_id, var2_id]], kdims=[var1_id, var2_id]).opts(
            color=self._colors['true_theta'], size=7, fontsize=self._FONTSIZES
        )

    def grand_data_plot(self, var_names: Iterable):
        plots = []
        cols = 3
        for i, var_id in enumerate(var_names):
            show_legend = True if i == cols - 1 else False
            postpred = self.density_plot(var_id, group='posterior_predictive')
            priopred = self.density_plot(var_id, group='prior_predictive')
            true = self.point_plot(var_id, what_var='data', what_point='true')
            map = self.point_plot(var_id, what_var='data', what_point='map')
            width = 600 if i % cols == cols - 1 else 400
            panel = (postpred * priopred * true * map).opts(
                legend_position='right', show_legend=show_legend, width=width, show_grid=True, fontsize=self._FONTSIZES,
                ylabel='',
            )
            plots.append(panel)

        return hv.Layout(plots).cols(cols)

    def grand_theta_plot(self, var1_id, var2_id, group='posterior'):
        plots = [
            self._plot_polytope_area(var1_id, var2_id),
            self._data_hull(var1_id=var1_id, var2_id=var2_id, group=group),
            self._bivariate_plot(var1_id=var1_id, var2_id=var2_id, group=group),
        ]
        if group == 'posterior':
            plots.extend([
                self.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='map'),
                self.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='true')
            ])
        return hv.Overlay(plots).opts(legend_position='right', show_legend=True, fontsize=self._FONTSIZES)


class MCMC_PLOT(PlotMonster):
    def __init__(
            self,
            polytope: LabellingPolytope,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None,
            hv_backend='bokeh',
    ):
        super().__init__(polytope, inference_data, v_rep, hv_backend)
        self._map = self._load_MAP()

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
                self._data['posterior_predictive']['data'].values[row, col, :], index=data_id
            ).T
            result['data']=data
        return result



class SMC_PLOT(PlotMonster):

    def __init__(
            self,
            polytope: LabellingPolytope,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None,
            hv_backend='bokeh',
    ):
        super().__init__(polytope, inference_data, v_rep, hv_backend)

    def plot_evolution(self, var1_id, var2_id=None, include_prior=True):
        plots = []
        if var2_id is None:
            num_chains = len(self._data['posterior']['chain'].values)
            addon = 1 if include_prior else 0
            for i in range(num_chains + addon):
                color = colorcet.glasbey[i]
                if i == 0:
                    color = self._colors['prior']
                    to_plot = self._get_chain(var1_id, chain_idx=0, group='prior')
                    label = 'prior'
                    muted = False
                else:
                    to_plot = self._get_chain(var1_id, chain_idx=i-1, group='posterior')
                    label = f'pop_{i}'
                    muted = True
                    if i == num_chains - 1:
                        muted = False
                        color = self._colors['posterior']
                        label = 'posterior'
                plots.append(
                    self._plot_distribution(to_plot, var_id=var1_id, label=label, color=color, muted=muted)
                )
        return hv.Overlay(plots).opts(legend_position='right', fontsize=self._FONTSIZES, **self._size_opts(width=600), )


def speed_plot():
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


def _load_data(hv_backend = 'matplotlib'):
    v_rep = pd.read_excel(r"C:\python_projects\sbmfi\v_rep.xlsx", index_col=0)
    pol = pickle.load(open(r"C:\python_projects\sbmfi\pol.p", 'rb'))

    mcmc_anton = az.from_netcdf(r"C:\python_projects\sbmfi\MCMC_e_coli_glc_anton_obsmod_wprior.nc")
    pmcmc_anton = MCMC_PLOT(pol, mcmc_anton, v_rep, hv_backend)

    mcmc_tomek = az.from_netcdf(r"C:\python_projects\sbmfi\MCMC_e_coli_glc_tomek_obsmod_CORR.nc")
    pmcmc_tomek = MCMC_PLOT(pol, mcmc_tomek, v_rep, hv_backend)

    smc_tomek = az.from_netcdf(r"C:\python_projects\sbmfi\SMC_e_coli_glc_tomek_obsmod_CORR_winv.nc")
    psmc_tomek = SMC_PLOT(pol, smc_tomek, v_rep, hv_backend)
    return pmcmc_anton, pmcmc_tomek, psmc_tomek

def MARGINAL_PLOT():
    variables = ['B_svd0', 'B_svd1', 'B_svd2', 'B_svd3', 'B_svd4', 'B_svd5', 'B_svd6',
                 'B_svd7', 'B_svd8', 'B_svd9', 'PGI_xch', 'FBA_xch', 'TPI_xch',
                 'GAPD_xch', 'PGK_xch', 'PGM_xch', 'ENO_xch', 'RPE_xch', 'RPI_xch',
                 'TKTa_xch', 'TKTb_xch', 'TKTc_xch', 'TALAa_xch', 'TALAb_xch',
                 'ACONTa_xch', 'ACONTb_xch', 'ICDHyr_xch', 'SUCOAS_xch', 'SUCDi_xch',
                 'FUM_xch', 'MDH_xch', 'PTAr_xch', 'ACKr_xch', 'GHMT2r_xch', 'GLYCL_xch']

    pmcmc_anton, pmcmc_tomek, psmc_tomek = _load_data()

    maxI = 10
    layout = []
    for i in range(maxI):
        varid = variables[i]

        dens_anton = pmcmc_anton.density_plot(var_id=varid)
        dens = dens_anton.data[('Distribution', 'Posterior')]
        dens.opts(color='#ffa500')
        dens_anton.data[('Distribution', 'Posterior')] = dens.relabel('Antoniewicz MCMC')

        dens_mcmc_tomek = pmcmc_tomek.density_plot(var_id=varid)
        dens = dens_mcmc_tomek.data[('Distribution', 'Posterior')]
        dens.opts(color='#b55ef1')
        dens_mcmc_tomek.data[('Distribution', 'Posterior')] = dens.relabel('LCMS MCMC')

        to_plot = psmc_tomek._get_chain(varid, chain_idx=-1)
        dens_smc_tomek = psmc_tomek._plot_distribution(to_plot, var_id=varid, label='LCMS SMC',
                                                       color=psmc_tomek._colors['posterior'])

        prior_dens = psmc_tomek.density_plot(var_id=varid, group='prior')

        true_plot = pmcmc_anton.point_plot(varid, label='Antoniewicz MLE')

        plot = (prior_dens * dens_anton * dens_mcmc_tomek * dens_smc_tomek * true_plot).opts(
            legend_position='right', **psmc_tomek._size_opts(width=600), fontsize=PlotMonster._FONTSIZES
        )
        if i != maxI - 1:
            plot.opts(show_legend=False)
        layout.append(plot)

    plot = hv.Layout(layout).cols(2).opts(hspace=0.1, vspace=0.2, fig_size=150)
    hv.save(plot, 'naks.png', fmt='png', dpi=400)


def PPC_PLOT():

    pmcmc_anton, pmcmc_tomek, psmc_tomek = _load_data()

    anton_ids = pd.Series(pmcmc_anton._data.posterior_predictive.data_id.values)
    tomek_ids = pd.Series(pmcmc_tomek._data.posterior_predictive.data_id.values)
    anton_ids = anton_ids.str.replace('_{M-}', '', regex=False)
    shared = anton_ids.loc[anton_ids.isin(tomek_ids)].values
    print(shared)
    varid = shared[1]
    print(varid)

    plot = pmcmc_anton.point_plot(varid, what_var='data')

    hv.save(plot, 'noks.png', fmt='png', dpi=400)





if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from bokeh.plotting import figure, output_file, save
    # model, kwargs = spiro(
    #     seed=None,
    #     batch_size=100,
    #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True,
    #     which_measurements='com', which_labellings=['A'], v5_reversible=True
    # )
    model, kwargs = spiro(
        seed=None,
        batch_size=10,
        backend='torch', v2_reversible=True, ratios=False, build_simulator=True, which_labellings=['A', 'B'],
        v5_reversible=True
    )
    spres = az.from_netcdf("C:\python_projects\sbmfi\src\sbmfi\inference\spiro_TEST.nc")
    v_rep = pd.read_excel('vrep.xlsx')
    mc = MCMC_PLOT(model._fcm.make_theta_polytope(), inference_data=spres, v_rep=v_rep)
    # R_svd0    R_svd1    R_svd2    R_svd3  v2_xch  v5_xch
    # aa = mc.point_plot('R_svd0', 'R_svd1')
    aa = mc.grand_theta_plot('R_svd0', 'v5_xch')
    # aa = mc.density_plot('R_svd3')
    # output_file(filename="custom_filename.html", title="plot2")
    show(hv.render(aa))
    # mc._v_rep.to_excel('vrep.xlsx', index=False)

