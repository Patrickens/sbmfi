# necessary to be able to plot stuff in arviz
from arviz.stats.density_utils import (
    _fast_kde_2d,
    kde,
    _find_hdi_contours,
    _get_bw,
    _get_grid,
    _kde_adaptive,
    _kde_convolution,
    histogram,
)
import arviz as az
import xarray as xr
import numpy as np
import warnings
from arviz.rcparams import rcParams
from arviz import InferenceData
from arviz.plots.plot_utils import default_grid, get_plotting_function
from arviz.stats.density_utils import _fast_kde_2d, kde, _find_hdi_contours
from arviz.plots.plot_utils import get_plotting_function, _init_kwargs_dict


bw = 'scott'


def plot_dist(
    values,
    values2=None,
    color="C0",
    kind="auto",
    cumulative=False,
    label=None,
    rotated=False,
    rug=False,
    bw=bw,
    quantiles=None,
    contour=True,
    fill_last=True,
    figsize=None,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    hist_kwargs=None,
    is_circular=False,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs,
):
    values = np.asarray(values)

    if isinstance(values, (InferenceData, xr.Dataset)):
        raise ValueError(
            "InferenceData or xarray.Dataset object detected,"
            " use plot_posterior, plot_density or plot_pair"
            " instead of plot_dist"
        )

    if kind not in ["auto", "kde", "hist"]:
        raise TypeError(f'Invalid "kind":{kind}. Select from {{"auto","kde","hist"}}')

    if kind == "auto":
        kind = "hist" if values.dtype.kind == "i" else rcParams["plot.density_kind"]

    dist_plot_args = dict(
        # User Facing API that can be simplified
        values=values,
        values2=values2,
        color=color,
        kind=kind,
        cumulative=cumulative,
        label=label,
        rotated=rotated,
        rug=rug,
        bw=bw,
        quantiles=quantiles,
        contour=contour,
        fill_last=fill_last,
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        hist_kwargs=hist_kwargs,
        ax=ax,
        backend_kwargs=backend_kwargs,
        is_circular=is_circular,
        show=show,
        **kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_dist", "distplot", backend)
    ax = plot(**dist_plot_args)
    return ax


def plot_kde(
    values,
    values2=None,
    cumulative=False,
    rug=False,
    label=None,
    bw=bw,
    adaptive=False,
    quantiles=None,
    rotated=False,
    contour=True,
    hdi_probs=None,
    fill_last=False,
    figsize=None,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    is_circular=False,
    ax=None,
    legend=True,
    backend=None,
    backend_kwargs=None,
    show=None,
    return_glyph=False,
    **kwargs
):
    if isinstance(values, xr.Dataset):
        raise ValueError(
            "Xarray dataset object detected. Use plot_posterior, plot_density "
            "or plot_pair instead of plot_kde"
        )
    if isinstance(values, InferenceData):
        raise ValueError(
            " Inference Data object detected. Use plot_posterior "
            "or plot_pair instead of plot_kde"
        )

    if values2 is None:

        if bw == "default":
            bw = "taylor" if is_circular else "experimental"

        grid, density = kde(values, is_circular, bw=bw, adaptive=adaptive, cumulative=cumulative)
        lower, upper = grid[0], grid[-1]

        density_q = density if cumulative else density.cumsum() / density.sum()

        # This is just a hack placeholder for now
        xmin, xmax, ymin, ymax, gridsize = [None] * 5
    else:
        gridsize = (128, 128) if contour else (256, 256)
        density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)

        if hdi_probs is not None:
            # Check hdi probs are within bounds (0, 1)
            if min(hdi_probs) <= 0 or max(hdi_probs) >= 1:
                raise ValueError("Highest density interval probabilities must be between 0 and 1")

            # Calculate contour levels and sort for matplotlib
            contour_levels = _find_hdi_contours(density, hdi_probs)
            contour_levels.sort()

            contour_level_list = [0] + list(contour_levels) + [density.max()]

            # Add keyword arguments to contour, contourf
            contour_kwargs = _init_kwargs_dict(contour_kwargs)
            if "levels" in contour_kwargs:
                warnings.warn(
                    "Both 'levels' in contour_kwargs and 'hdi_probs' have been specified."
                    "Using 'hdi_probs' in favor of 'levels'.",
                    UserWarning,
                )
            contour_kwargs["levels"] = contour_level_list

            contourf_kwargs = _init_kwargs_dict(contourf_kwargs)
            if "levels" in contourf_kwargs:
                warnings.warn(
                    "Both 'levels' in contourf_kwargs and 'hdi_probs' have been specified."
                    "Using 'hdi_probs' in favor of 'levels'.",
                    UserWarning,
                )
            contourf_kwargs["levels"] = contour_level_list

        lower, upper, density_q = [None] * 3

    kde_plot_args = dict(
        # Internal API
        density=density,
        lower=lower,
        upper=upper,
        density_q=density_q,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        gridsize=gridsize,
        # User Facing API that can be simplified
        values=values,
        values2=values2,
        rug=rug,
        label=label,
        quantiles=quantiles,
        rotated=rotated,
        contour=contour,
        fill_last=fill_last,
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        is_circular=is_circular,
        ax=ax,
        legend=legend,
        backend_kwargs=backend_kwargs,
        show=show,
        return_glyph=return_glyph,
        **kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_kde", "kdeplot", backend)
    ax = plot(**kde_plot_args)

    return ax


def _kde_linear(
    x,
    bw=bw,
    adaptive=False,
    extend=False,
    bound_correction=True,
    extend_fct=0,
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
    cumulative=False,
    grid_len=512,
    **kwargs,  # pylint: disable=unused-argument
):
    # Check `bw_fct` is numeric and positive
    if not isinstance(bw_fct, (int, float, np.integer, np.floating)):
        raise TypeError(f"`bw_fct` must be a positive number, not an object of {type(bw_fct)}.")

    if bw_fct <= 0:
        raise ValueError(f"`bw_fct` must be a positive number, not {bw_fct}.")

    # Preliminary calculations
    x_min = x.min()
    x_max = x.max()
    x_std = np.std(x)
    x_range = x_max - x_min

    # Determine grid
    grid_min, grid_max, grid_len = _get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend, bound_correction
    )
    grid_counts, _, grid_edges = histogram(x, grid_len, (grid_min, grid_max))

    # Bandwidth estimation
    bw = bw_fct * _get_bw(x, bw, grid_counts, x_std, x_range)

    # Density estimation
    if adaptive:
        grid, pdf = _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    else:
        grid, pdf = _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction)

    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf

az.plots.distplot.plot_dist = plot_dist
az.plots.kdeplot.plot_kde = plot_kde
az.stats.density_utils._kde_linear = _kde_linear
