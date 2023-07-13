import pandas as pd
import numpy as np
import re
import copy
from pathlib import Path
from scipy.stats import linregress

"""
Since I will be using the tecan sunrise for all OD measurements, it is useful to have some standardized scripts
to calculate OD and growth-rates automatically from a folder of excel files
"""

def zero_time(ts, convert=True):
    T0 = copy.deepcopy(ts['T0'])
    for i, col in ts.iteritems():
        if re.search('(T[0-9]+)',i):
            if convert:
                ts[i] = (col-T0).total_seconds()/(60*60)
            else:
                ts[i] = col-T0
    return ts.astype(float)

def format_od(path_str):
    vals = pd.read_excel(io=path_str, skiprows=10, nrows=8, index_col=0).loc[:, :12].reset_index().melt(id_vars=['<>']).rename({'<>': 'row', 'value': 'od'}, axis=1)
    vals['well'] = vals['row'] + vals['variable'].astype(str)
    vals = vals.loc[:,['well', 'od']].set_index('well')
    return vals.astype(float)

def read_date_time(path_str):
    df = pd.read_excel(io=path_str, skiprows=0, usecols=[5], nrows=2)
    date = df.iloc[0,0]#.to_pydatetime()
    time = df.iloc[1,0]
    date = date.replace(hour=time.hour, minute=time.minute, second=time.second)
    return date

def linear_fit_sm(x, y):
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y,X,missing='drop').fit()
        return pd.Series(np.concatenate((model.params,[model.rsquared])), index=['b0','a','r2'])
    except BaseException:
        return pd.Series(np.concatenate([np.nan, np.nan, np.nan]), index=['b0','a','r2'])

def linear_fit_sp(x, y):
    with np.errstate(invalid='ignore'):
        try:
            # this is if we filter the OD values to exclude the log/ stationary phase by setting them to np.nan
            mask = ~np.isnan(x) & ~np.isnan(y)
            slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
        except BaseException:
            intercept, slope, r_value = np.nan, np.nan, np.nan
        return pd.Series([intercept, slope, r_value], index=['b0','a','r2'])

def make_tecan_df(folder, sep='_'):
    """
    File naming convention is that there is T{int} in the name, and that everything is separated by sep
    """
    pathlist = Path(folder).glob('**/*.xlsx')
    val_dct = {}
    time_dct = {}
    for i, path in enumerate(pathlist):
        path_str = str(path)
        match = re.search(f'([{sep}]?T\d+[{sep}]?)', path.stem)
        if match:
            T = match.group()
            datetime = read_date_time(path_str=path_str)
            if (match.start() == 0) or (T[-1] != sep):
                group = path.stem.replace(T, '')
            else:
                group = path.stem.replace(T, sep)

            T = T.replace(sep,'')
            time_store = time_dct.setdefault(T, {})
            time_store[group] = datetime

            val_store = val_dct.setdefault(group, {})
            val_store[T] = format_od(path_str)['od']


    res = {}
    for group, dct in val_dct.items():
        res[group] = pd.DataFrame(dct)

    return res, pd.DataFrame(time_dct)

def calculate_od(abs_df_dct, blank, dilution=4, C=2.3):
    if isinstance(blank, dict):
        # NOTE: this is to have a separate blank for every sample
        raise NotImplementedError
    od_dct = {}
    for group, df in abs_df_dct.items():
        od_dct[group] = df.sub(blank.values, axis=0) * dilution * C
    return od_dct

def concat(dct):
    for group, df in dct.items():
        df.insert(0, 'group', group)
    return pd.concat(dct.values())

def mutilation(odf_dct, tdf):
    mu_dct = {}
    for group, odf in odf_dct.items():
        ts = zero_time(ts=tdf.loc[group, :]).dropna()
        with np.errstate(invalid='ignore'):
            mu_dct[group] = np.log(odf).apply(lambda row: linear_fit_sp(ts.values, row.values), axis=1)
    return mu_dct

def make_tecan_nano():
    pass



if __name__ == "__main__":
    import datetime
    folder = r"C:\python_projects\pysumo\measurement_model\TFGhent T8 T12"
    blank = r"C:\python_projects\pysumo\measurement_model\TFGhent T8 T12\BLANK.xlsx"
    blank = format_od(blank)
    abs_df_dct, tdf = make_tecan_df(folder=folder)
    od_dct = calculate_od(abs_df_dct=abs_df_dct, blank=blank['od'])
    mu_dct = mutilation(odf_dct=od_dct, tdf=tdf)

    eedf = pd.read_excel(r"C:\python_projects\pysumo\measurement_model\tdf.xlsx", index_col=0).drop(['TI'], axis=1)
    edf = eedf.vecopy()
    for i, row in edf.iterrows():
        T0 = copy.deepcopy(row['T0'])
        for j, col in row.iteritems():
            if re.search('(T[H]?[0-9]+)', j):
                edf.loc[i, j] = (col - T0).total_seconds() / (60 * 60)

    for group, df in od_dct.items():
        times = edf.loc[group, :].dropna()
        for i, (T, time) in enumerate(times.iteritems()):
            if T in df.columns:
                continue
            mudf = mu_dct[group]
            df.insert(i, T, np.exp(time * mudf['a'] + mudf['b0']))

    mdf = concat(od_dct)
    mudf = concat(mu_dct)
    adf = concat(abs_df_dct)
    edf.index.name = 'group'
    eedf.index.name = 'group'
    with pd.ExcelWriter('TFGhent_physiology.xlsx') as writer:
        mdf.to_excel(writer, sheet_name='od600')
        mudf.to_excel(writer, sheet_name='mu')
        edf.to_excel(writer, sheet_name='time')
        eedf.to_excel(writer, sheet_name='time_raw')
        adf.to_excel(writer, sheet_name='absorbance')
        blank.to_excel(writer, sheet_name='blank')