import numpy as np
import pandas as pd
from sbmfi.compound.formula import Formula
from sbmfi.compound.adducts import emzed_adducts
import pickle
import os
from sbmfi.settings import BASE_DIR
# import emzed
# from emzed.quantification.peak_shape_models.simplified_emg import SimplifiedEmgModel
import multiprocessing as mp
import copy
from typing import Iterable
import scipy

_adduct_cols = ['m_multiplier', 'adduct_add', 'adduct_sub', 'sign_z', 'z']
_integration_cols = ['peak_shape_model', 'area', 'rmse', 'valid_model']
_table_cols = ['bigg_id', 'n_isomers', 'mf', 'adduct_name', 'nC13', 'isotope_decomposition', 'mz', 'mzmin', 'mzmax', 'rt', 'rtmin', 'rtmax']


def add_formulas(peak_df, drop=True, keepz=False, abundance=False, ppm=10.0):
    peak_df = peak_df.reset_index(drop=True)
    peak_df['isotope_decomposition'] = 'nan'
    peak_df['mz'] = 0.0
    ision = 'adduct_name' in peak_df

    if abundance:
        peak_df['abundance'] = 0.0

    for i, row in peak_df.iterrows():
        f = Formula(row['mf'])
        if ision:
            f = f * row['m_multiplier'] \
                + Formula(row['adduct_add']) \
                - Formula(row['adduct_sub']) \
                + {'-': row['z'] * -row['sign_z']}
        if 'nC13' in row:
            f = f.add_C13(row['nC13'])
        peak_df.loc[i, 'isotope_decomposition'] = f.to_chnops()
        peak_df.loc[i, 'mz'] = f.mass(ion=ision)
        if abundance:
            peak_df.loc[i, 'abundance'] = f.abundance()

    if ppm > 0.0:
        shift = ((ppm / 1e6) * peak_df['mz']) / 2
        peak_df['mzmin'] = peak_df['mz'] - shift
        peak_df['mzmax'] = peak_df['mz'] + shift

    if drop:
        slicer = None
        if keepz:
            slicer = -1
        peak_df.drop(_adduct_cols[slice(None, slicer)], axis=1, inplace=True)
    return peak_df


def extract_model_slots(model, attr_name='_INTENS'):
    return getattr(model, attr_name)


def emg_rtstats(model, boundmin=20, boundfrac=0.05):  # rt and mode are pretty much the same for all peaks
    if not model.is_valid or (model.model_name != 'emg'):
        return np.zeros(3)

    rts = np.linspace(model._rtmin, model._rtmax, model.NPOINTS_GRAPH)
    ii_full = SimplifiedEmgModel._apply(rts, model._height, model._center, model._width, model._symmetry)

    if (ii_full < boundmin).all():
        apex = ((model._rtmax - model._rtmin) / 2.5) + model._rtmin
        return np.array([model._rtmin, apex, model._rtmax])

    apex_idx = ii_full.argmax()
    bound = max(boundmin, boundfrac * ii_full[apex_idx])

    if (apex_idx == 0) and (
            model._rtmin < 2.0):  # this is necessary for when the first or last item is already larger than bounds
        apex_idx = 1
    elif (apex_idx == ii_full.shape[0]) and (model._rtmax > 1000):
        apex_idx = ii_full.shape[0] - 1

    if ii_full[0] > bound:
        ii_full[0] = 0.0
    if ii_full[-1] > bound:
        ii_full[-1] = 0.0

    left_idx = np.where(ii_full[:apex_idx] < bound)[0]
    right_idx = np.where(ii_full[apex_idx:] < bound)[0]

    if (left_idx.size > 0) and (right_idx.size > 0):
        left_idx = left_idx[-1]
        right_idx = right_idx[0] + apex_idx
    else:
        copmod = copy.copy(model)
        copmod.NPOINTS_GRAPH += 100
        if copmod._rtmin > 2.0:
            copmod._rtmin -= 2.0
        if copmod._rtmax < 1000:  # WATCH OUT WITH LONGER GRADIENTS WITH RT_MAX > 1000s
            copmod._rtmax += 2.0
        return emg_rtstats(model=copmod, boundmin=boundmin, boundfrac=boundfrac)
    return rts[[left_idx, apex_idx, right_idx]]


def hedgehoggin(model, rtmin, rtmax):
    rts = model._RTS
    where = (rts > rtmin) & (rts < rtmax)
    valid = np.where(where)[0]
    intens = model._INTENS[valid]
    denom = intens.shape[0]
    if denom <= 2:
        return 0.0, denom, 1.0

    where = ~(where & (model._INTENS < 1.0))
    valid = np.where(where)[0]
    model = SimplifiedEmgModel._fit(rts[valid], model._INTENS[valid], {})  # this is a refitted area where 0s are replaced by nans!
    area = 1.0
    if model.is_valid:
        area = model.area
    return (intens > 1.0).sum() / denom, denom, area


def correlate_scans_linear(df, adduct=True, minM=True):  # TODO: do same for isotopes!
    if (df.shape[0] == 1) or df['intensities'].isna().any():  # if only M-H or M+0 dont correlate
        return df

    df = df.sort_values('nC13', ascending=minM)
    intensities = np.stack(df['intensities'].values)
    if adduct:
        corr = pd.DataFrame(intensities, index=df['adduct_name']).T.corr().loc[:, 'M-H']
    else:
        # log-correlate isotopomers
        corr = pd.DataFrame(np.log(intensities + 1.0), index=df['nC13']).T.corr().iloc[0]
    try:
        df.loc[:, 'scancorr'] = corr.values
    except:
        pass
        # print(corr)
    return df


def extract_scan_stats(integrand, boundfrac=0.1, boundmin=20, minM=True, correlate=False, drop_intensities=True):
    # this function fits (max. lik.) exponentially modified gaussian to determine the apex, start and end of the peak

    if len(set(_integration_cols).intersection(set(integrand.col_names))) != 4:
        raise ValueError('please integrate emg for all peaks in table')
    psms = set(integrand.peak_shape_model.to_list())
    if (len(psms) > 1) or (list(psms)[0] != 'emg'):
        raise ValueError('stats only work for EMG peaks')

    # peak shape stats
    models = integrand['model'].to_list()
    bounds = np.stack(list(map(lambda m: emg_rtstats(m, boundmin=boundmin, boundfrac=boundfrac), models)))
    hedgehog = np.stack(list(map(lambda args: hedgehoggin(*args), zip(models, bounds[:, 0], bounds[:, 2]))))

    # linear integration
    linarea = np.fromiter(map(lambda m: np.trapz(m._INTENS, m._RTS), models), dtype=float)[:, None]

    # peak statistics and lin_area
    rtresult = pd.DataFrame(np.hstack([bounds, hedgehog, linarea]),
                            columns=['emg_min', 'emg_apex', 'emg_max', 'hedgehog', 'nscans', 'nonan_area', 'lin_area'])
    rtresult['nscans'] = rtresult['nscans'].astype(int)
    # maybe add a peak counting statistic?

    # correlations
    # this adds a numpy array with intensities to the table
    integrand.add_or_replace_column("intensities", integrand.apply(extract_model_slots, integrand.model), object)
    coresults = integrand.to_pandas()
    integrand.drop_columns('intensities')

    gbyadd = ['bigg_id', 'adduct_name']
    gbylab = ['bigg_id', 'nC13']
    if 'source' in coresults.columns:
        gbyadd = ['source'] + gbyadd
        gbylab = ['source'] + gbylab

    coresults = pd.concat([coresults, rtresult], axis=1)
    if correlate:
        coresults['scancorr'] = 1.0
        if 'abundance' in coresults.columns:
            coresults['theoratio'] = 1.0

        coresults = coresults.groupby(gbyadd).apply(lambda df: correlate_scans_linear(df, adduct=False, minM=minM))
        if isinstance(coresults.index, pd.MultiIndex):
            coresults.index = coresults.index.get_level_values(-1)
        coresults = coresults.sort_index()
        coresults = coresults.groupby(gbylab, as_index=True).apply(lambda df: correlate_scans_linear(df))
    if drop_intensities:
        coresults = coresults.drop('intensities', axis=1)
    return coresults


def _init_worker(table_df: pd.DataFrame, stat_kwargs: dict={}):
    global TABLE, STATKWARGS
    TABLE = emzed.table.Table.from_pandas(table_df)
    STATKWARGS = stat_kwargs

def _worker(path):
    global TABLE
    table = TABLE.copy()
    try:
        peakmap = emzed.io.load_peak_map(path)
        table.add_column_with_constant_value('peakmap', peakmap, emzed.PeakMap)
        table.add_or_replace_column('source', table.apply(lambda v: v.meta_data['source'], table.peakmap), str)
        integrand = emzed.quantification.integrate(table, "emg", ms_level=1)
        table.close()
        stats = extract_scan_stats(integrand, **STATKWARGS)
        peakmap.close()
        integrand.close()
    except:
        stats = None
        print(path)
    return stats

def process_tasks(
        table_df: pd.DataFrame, tasks: Iterable, num_processes=0, boundfrac=0.1, boundmin=20, correlate=False, minM=True
):
    stat_kwargs = dict(boundfrac=boundfrac, boundmin=boundmin, correlate=correlate, minM=minM)
    if num_processes == 0:
        _init_worker(table_df=table_df, stat_kwargs=stat_kwargs)
        result = []
        for path in tasks:
            result.append(_worker(path=path))
    else:
        pool = mp.Pool(processes=num_processes, initializer=_init_worker, initargs=(table_df, stat_kwargs))
        result = pool.map(_worker, iterable=tasks)
        pool.close()
    return pd.concat(result, axis=0)


def compute_abundances(isodecomp: np.array, max_C=2):
    U13Cglucose_abundances = copy.deepcopy(_nist_mass)
    U13Cglucose_abundances['C'][0] = (13.0033548378, 1.0)
    U13Cglucose_abundances['C'][12] = (12.0, 0.01)
    U13Cglucose_abundances['C'][13] = (13.0033548378, 0.99)

    mapper = {}

    C_in = np.arange(max_C)
    for ic in isodecomp:
        fic = Formula(ic)
        f = fic.no_isotope()
        if f['C'] > 2:
            if fic['[12]C'] in f['C'] - C_in:
                ab = fic.abundance()
            elif fic['[13]C'] in f['C'] - C_in:
                ab = fic.abundance(U13Cglucose_abundances)
        else:
            ab = 0.0
        mapper[ic] = ab

    return mapper


def make_result(
        data: pd.DataFrame,
        what_values = 'area',  # area or lin_area
        min_periods = 10,
):
    # data = data.loc[data['bigg_id'] == 'gal']

    data = data.sort_values(by=['bigg_id', 'adduct_name', 'nC13'], ascending=True)
    results = {}
    for i, ((mixf, dil), df) in enumerate(data.groupby(['mix_frac', 'dil'])):
        original = df.pivot(
            index=['mf', 'nC', 'bigg_id', 'adduct_name', 'nC', 'nC13', 'abundance', 'theor'],
            columns=['inj'], values=[what_values]
        )
        pivoted = np.log10(original + 1.0)

        retention_times = df.pivot(
            index=['mf', 'nC', 'bigg_id', 'adduct_name', 'nC', 'nC13', 'abundance', 'theor'],
            columns=['inj'], values=['emg_apex']
        )

        index = pivoted.index.to_frame(index=False)

        mu = pivoted.mean(1).values
        std = pivoted.std(1).values

        rt_mu = retention_times.mean(1).values
        rt_std = retention_times.std(1).values
        rt_corr = retention_times.T.corr(min_periods=min_periods).values

        triurow, triucol = np.triu_indices(pivoted.shape[0])

        for j, (rowa, rowb) in enumerate(zip(triurow, triucol)):
            x = pivoted.values[rowa, :]
            y = pivoted.values[rowb, :]
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]

            if (x.shape[0] > min_periods) and (rowa != rowb):
                alpha, beta, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                # NOTE: R_sqrd = 1 - (SS_res/SS_tot) and rho = sqrt(R_sqrd), only holds for linear models
                # SS_res = ((y - (alpha*x + beta))**2).mean()
                # SS_tot = ((y - y.mean()) ** 2).sum()

                # TODO must be a more efficient way of unpacking....
                theor = index['theor'][rowa]
                nC = index['nC'][rowa]
                bigg_a, bigg_b = index['bigg_id'][rowa], index['bigg_id'][rowb]
                add_a, add_b = index['adduct_name'][rowa], index['adduct_name'][rowb]
                nC13_a, nC13_b = index['nC13'][rowa], index['nC13'][rowb]
                ab_a, ab_b = index['abundance'][rowa], index['abundance'][rowb]

                result = {
                    'mf_a': index['mf'][rowa], 'mf_b': index['mf'][rowb],
                    'nC_a': index['nC'][rowa], 'nC_b': index['nC'][rowb],
                    'ab_a': ab_a, 'ab_b': ab_b,

                    'theor': np.nan,
                    'alpha': alpha, 'beta': beta, 'corr': r_value,
                    'mu_a': mu[rowa], 'mu_b': mu[rowb],
                    'std_a': std[rowa], 'std_b': std[rowb],

                    'rt_a': rt_mu[rowa], 'rt_b': rt_mu[rowb],
                    'std_rt_a': rt_std[rowa], 'std_rt_b': rt_std[rowb],
                    'rt_corr': rt_corr[rowa, rowb],
                }

                natab = (nC13_a == 0) & (nC13_b == 1)
                impur = (nC13_a == nC - 1) & (nC13_b == nC)
                mix   = (nC13_a == 0) & (nC13_b == nC)
                if ((bigg_a == bigg_b) & (add_a == add_b)) and (natab or impur or mix):
                    if nC < 3:
                        # this is because we otherwise get large interference in the M+1 signal from both the narually labelled and fully labelled sample
                        continue

                    xo = original.values[rowa, mask]
                    yo = original.values[rowb, mask]
                    fracs = yo / (xo + yo)
                    frac = fracs.mean()

                    if natab or impur:
                        compare = ab_b / (ab_a + ab_b)
                        comparison = 'natab' if natab else 'impur'
                    if mix:
                        compare = theor
                        comparison = 'mix'

                    result = {
                        **result, 'frac': frac, 'theor': compare, 'bias': frac - compare, 'comparison': comparison,
                    }

                results[(bigg_a, bigg_b, add_a, add_b, nC13_a, nC13_b, mixf, dil)] = result

    results = pd.DataFrame(results).T
    results.index = results.index.rename(
        names=['bigg_a', 'bigg_b', 'add_a', 'add_b', 'nC13_a', 'nC13_b', 'mix_frac', 'dil']
    )
    results = results.reset_index()
    return results


def mangle_results(results):
    selector = results['mu_a'] < results['mu_b']
    acols = results.columns[results.columns.str.contains('_a$')]
    bcols = acols.str.rstrip('_a') + '_b'
    remember = results.loc[selector, acols].copy()
    results.loc[selector, acols] = results.loc[selector, bcols].values
    results.loc[selector, bcols] = remember.values

    results.loc[selector, 'frac'] = 1.0 - results.loc[selector, 'frac']
    results.loc[selector, 'theor'] = 1.0 - results.loc[selector, 'theor']
    results.loc[selector, 'bias'] *= -1

    results = results.astype({'nC13_a': int, 'nC13_b': int})  # ??????????????????????
    return results


if __name__ == "__main__":

    data = pd.read_csv(r"C:\python_projects\pysumo\raw_data_processing\chapobsmod\ALL_RATIO_DATA_FILTERED_20PPM.csv")

    res = make_result(data)
    res.to_csv('AREA_RESULT_20PPM_NEW.csv', index=False)

    # DATA_FOLDER = 'X:/sauer_1/~archive/Tomek/6546'
    # table_df = pd.read_excel(r"C:\python_projects\pysumo\raw_data_processing\chapobsmod\IJO_COLIOME_20PPM_CURATED_MASISOS.xlsx")
    # observation_folder = os.path.join(DATA_FOLDER, '20220209 LCMS observation model', 'mzml')
    #
    # tasks = []
    # for file in os.listdir(observation_folder):
    #     # print('mix_dil' not in file, 'pure_dil' not in file)
    #     if ('mix_dil' not in file) and ('pure_dil' not in file):
    #         continue
    #     tasks.append(os.path.join(observation_folder, file))
    #
    # ALL = process_tasks(table_df, tasks, num_processes=5)
    # pickle.dump(ALL, open('ALL_RATIO_DATA_20PPM.p', 'wb'))
    # ALL.to_csv('ALL_RATIO_DATA_20PPM.csv', index=False)