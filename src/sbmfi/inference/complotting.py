from sbmfi.core.polytopia import PolytopeSamplingModel, V_representation, fast_FVA, LabellingPolytope
import arviz as az
import pandas as pd
import holoviews as hv
from scipy.spatial import ConvexHull
import numpy as np
from typing import Iterable, Union, Dict, Tuple


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
            'posterior_predictive': 'data',
            'prior_predictive': 'data',
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
                self._data['posterior_predictive']['data'].values[row, col, :], index=data_id
            ).T
            result['data']=data
        return result

    def _load_true_theta(self):
        theta_id = self._data.posterior['theta_id'].values
        true_theta = self._data.attrs.get('true_theta')
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


if __name__ == "__main__":
    import pickle, os