import numpy as np
import pandas as pd
import scipy


def compute_correlationstats(groups, df, to_extract='area'):

    pivdf = df.pivot(index=['source'], columns=['nC13'], values=['area', 'lin_area']).sort_index(axis=1, level=1)
    areadf = pivdf.loc[:, (to_extract, slice(None))]
    areadf.columns = areadf.columns.get_level_values(-1)

    triurow, triucol = np.triu_indices(areadf.shape[1])

    mu = areadf.mean()
    std = areadf.std()
    cov = areadf.cov()
    cor = areadf.corr()
    # print(scipy.stats._multivariate._PSD(np.sqrt(areadf.cov().values)))
    try:
        loglik = np.nansum(scipy.stats.multivariate_normal.logpdf(x=areadf.values, mean=mu.values, cov=cov.values))
    except:
        loglik = np.nan

    cov = pd.DataFrame({
        **groups,
        'nC13_a': areadf.columns[triurow],
        'nC13_b': areadf.columns[triucol],
        'cov': cov.values[triurow, triucol],
        'cor': cor.values[triurow, triucol],
    })

    mu = pd.DataFrame({
        **groups,
        'nC13': mu.index,
        'mu': mu.values,
        'std': std.values,
    })

    loglik = pd.Series({
        **groups,
        'loglik': loglik,
    })
    return cov, mu, loglik


def extract_correlationstats(gby, merge=True, sort=True):
    results = []
    for i, (groups, df) in enumerate(gby):
        groups = dict(zip(gby.keys, groups))
        results.append(compute_correlationstats(groups, df))

    results = np.array(results, dtype=object).T
    cov = pd.concat(results[0])
    mu = pd.concat(results[1])
    loglik = pd.DataFrame(list(results[2]))

    if not merge:
        return cov, mu, loglik

    temp = cov.merge(mu, left_on=['exp', 'bigg_id', 'adduct_name', 'nC13_a'], right_on=['exp', 'bigg_id', 'adduct_name', 'nC13']).drop('nC13', axis=1)
    result = temp.merge(mu, left_on=['exp', 'bigg_id', 'adduct_name', 'nC13_b'], right_on=['exp', 'bigg_id', 'adduct_name', 'nC13'], suffixes=['_a', '_b']).drop('nC13', axis=1)
    selector = result['mu_a'] < result['mu_b']
    remember = result.loc[selector, ['nC13_a', 'mu_a', 'std_a']].copy()
    result.loc[selector, ['nC13_a', 'mu_a', 'std_a']] = result.loc[selector, ['nC13_b', 'mu_b', 'std_b']].values
    result.loc[selector, ['nC13_b', 'mu_b', 'std_b']] = remember.values
    result = result.astype({'nC13_a': int, 'nC13_b': int})
    return result, loglik