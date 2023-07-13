import pandas as pd
import numpy as np
from pathlib import Path, PureWindowsPath
import os, re, json
from sbmfi.lcmsanalysis.formula import Formula


def sequencer_source_file_processer(df, randomize=None):
    # TODO: randomize within group; do not shuffle multiple injections
    cols = {
        'group': 0,
        'id': 'A1', # this is the same as sample name!
        'plate': 1,
        'pos': 'A1',
        'amount': 0.0,
        'growth': 0.0,
        'factor:A': '...',
        'factor:B': '...',
        'method': 'A',
        'n_injections': 1,
    }

    # NOTE: check whether all columns are there, line below does not work due to variable factor:s
    # df = df.loc[:, list(cols.keys())]
    if randomize is not None:
        raise NotImplementedError

    if 'n_injections' in df.columns:
        ndf = df.loc[df.index.repeat(df['n_injections'])].reset_index(drop=True)
        ndf['injection'] = 0
        for a, gdf in ndf.groupby(['id', 'method']):
            ndf.loc[gdf.index, 'injection'] = range(gdf.shape[0])
    else:
        ndf = df

    # NOTE: this is only necessary for fole names; keep it for the ids! makes data-analysis easier
    id_to_name = ndf['id'].str.replace('[<>:"\/|?*]', '_', regex=True)

    if not id_to_name.str.count('_').unique().shape[0] == 1:
        print('WATCH OUT: file names have different number of underscores!')

    if ('method' not in ndf.columns) or (ndf['method'].unique().shape[0] == 1):
        ndf['sample_name'] = id_to_name
    else:
        ndf['sample_name'] = id_to_name + '_' + ndf['method']

    ndf['file_name'] = ndf['sample_name']
    if 'injection' in ndf.columns:
        ndf['sample_name'] += '_inj' + ndf['injection'].astype(str)


    # NOTE: this is only for Riekes file! Make this more general!!!
    ndf = ndf.reset_index(drop=True)
    map = {}
    for plate, df in ndf.groupby('plate'):
        index = df.loc[df['group'] == 'sample'].index.values
        shuffled = index.copy()
        np.random.shuffle(shuffled)
        map.update(dict(zip(index, shuffled)))
    for i in range(ndf.shape[0]):
        if i not in map:
            map[i] = i
    ndf.index = ndf.index.map(map)
    ndf = ndf.loc[range(ndf.shape[0])]
    # ndf.to_excel(r"C:\python_projects\pysumo\test.xlsx", index=False)
    return ndf


def qe_worklist(df):
    # save as csv
    default_method = {'A': r'C:\Xcalibur\methods\20201030_Dilution_MDV_1e6AGC_100ms_IT_FULL_MS1.meth'}
    A1_cell = 'Bracket Type=4'
    cols = {
        'Sample Type': 'Unknown',
        'File Name': 'SUMO_MM2_data_00001', # has to be unique
        'Sample ID': 'sample',
        'Position': '1:A,1', # weird position format
        'Path': r'D:\Users\tomekd\SUMO_MM2', # data folder
        'Instrument Method': (r'C:\Xcalibur\methods\20201030_Dilution_MDV_1e6AGC_100ms_IT_FULL_MS1.meth', None),
        'Inj Vol': (2.0, None),
        'Sample Name': 'SUMO_MM2_01',
    }
    # TODO: map the ndf columns to the QE columns and create file

    ndf = sequencer_source_file_processer(df=df)


def openbis_registration_file(df):
    # save as tsv
    cols = {
        'SAMPLEPOSITION': 'P01-A1',
        'SAMPLENAME': 'SUMO_MM2_01',
        'SAMPLEPERTURBATION': '-1xM | 00000000 | 00',
        'SAMPLETIME': None,
        'SAMPLEAMOUNT': None,
        'SAMPLESPECIES': None,
    }
    ndf = sequencer_source_file_processer(df=df)
    ndf = ndf.drop_duplicates(subset=['sample_name'], keep='first')
    data = {}
    for col_name, example in cols.items():
        if col_name == 'SAMPLEPOSITION':
            column = 'P' + ndf['plate'].astype(str) + '-' + ndf['pos']
        elif col_name == 'SAMPLENAME':
            column = ndf['sample_name']
        elif col_name == 'SAMPLEPERTURBATION':
            column = ndf.filter(regex='factor:', axis=1).astype(str).apply(lambda x: ' | '.join(x), axis=1)
        elif col_name == 'SAMPLETIME':
            column = None
        elif col_name == 'SAMPLEAMOUNT':
            if 'amount' in ndf.columns:
                column = ndf['amount']
            else:
                column = None
        elif col_name == 'SAMPLESPECIES':
            if 'growth' in ndf.columns:
                column = ndf['growth']
            else:
                column = None
        data[col_name] = column

    return pd.DataFrame(data)


def tof6546_worklist(df, project_name, data_folder=r'D:\Projects\aothman\Data\tomek', method_map=None, sample=True):
    # save as csv
    if method_map is None:
        method_map = {'A':  r'D:\MassHunter\Methods\_FIA\fia_1290_neg_single.m'}
    cols = {
        'Sample Index': 1, # needs to be unique
        'Sample Name': 'SUMO_MM2_01', # this can be the same for many injections of the same well/sample
        'Sample Position': 'P1-A1',
        'Data File': r'D:\MassHunter\data\tomekd\SUMO_error_model\SUMO_error_model_0001.d', # needs to be unique for every injection
        'Method': (r'D:\MassHunter\Data\All\method repository\fia_ctc_neg_AF_double.m', None),
        'Sample Type': 'Sample',
        'Inj Vol': (2.0, 'As Method'),
        'Comment': 'perturbation',
    }
    # TODO: map the ndf columns to the 6546 columns and create file

    ndf = sequencer_source_file_processer(df=df)

    data = {}
    for col_name, example in cols.items():
        if col_name == 'Sample Index':
            column = range(ndf.shape[0])
        elif col_name == 'Sample Name':
            column = ndf['sample_name']
        elif col_name == 'Sample Position':
            column = 'P' + ndf['plate'].astype(str) + '-' + ndf['pos']
        elif col_name == 'Data File':
            if sample:
                ding = ndf['sample_name']
            else:
                ding = ndf['file_name']
            column = data_folder + '\\' + project_name + '\\' + ding + '.d'
        elif col_name == 'Method':
            column = ndf['method'].map(method_map)
        elif col_name == 'Sample Type':
            column = 'Sample'
        elif col_name == 'Inj Vol':
            column = 'As Method'
        elif col_name == 'Comment':
            column = ndf['id']

        data[col_name] = column
    return pd.DataFrame(data)


def openbis_upload(df, tsv_file, project_name, double_inj=False):
    # NOTE: first line of file should be ##diederen@imsb.biol.ethz.ch

    cols = {
        'file_name': '',
        'sample': '',
        'experiment': '',
        'project': '',
        'space': '',
        'conversion': '',
        'datasetcomments': '',
    }
    ndf = sequencer_source_file_processer(df=df)
    sample_ids = pd.read_csv(tsv_file, sep='\t')
    sample_map = pd.Series(sample_ids['Code'].values, index=sample_ids['Name']).to_dict()

    data = {}
    for col_name, example in cols.items():
        if col_name == 'file_name':
            column = ndf['file_name'] + '.fiaML' # NOTE: this is because 'a' is automatically appended ffs
        elif col_name == 'sample':
            column = ndf['sample_name'].map(sample_map)
        elif col_name == 'experiment':
            column = None
        elif col_name == 'project':
            column = None
        elif col_name == 'space':
            column = 'ECOLI'
        elif col_name == 'conversion':
            column = None
        elif col_name == 'datasetcomments':
            column = ndf.loc[:, ['plate']].apply(lambda x: f"P{x['plate']:03d}:{x.name:03d} {project_name}", axis=1)

        data[col_name] = column
    uploadf = pd.DataFrame(data)

    if double_inj:
        inj_a = uploadf.copy()
        inj_b = uploadf.copy()
        inj_a['file_name'] = inj_a['file_name'].str.replace('.fiaML$', 'a.fiaMl', regex=True)
        inj_b['file_name'] = inj_b['file_name'].str.replace('.fiaML$', 'b.fiaMl', regex=True)
        uploadf = pd.concat([inj_a, inj_b], axis=0).sort_values('file_name')
    return uploadf


def chrominer_json(adf, folder, CIso=None, mztol=0.005):
    ### NOTE: this only works when not running with admin rights!
    ###     e.g. folder = r'\\d.ethz.ch\groups\biol\sysbc\sauer_1\users\Tomek\ALAA LC-MS first file\MM_FormicAcid_only_test'
    # TODO: need to remove the sauer1 file-path from the files; crunch assumes everything is already on sauer1
    #   remove \\d.ethz.ch\groups\biol\sysbc\sauer_1
    chromin = {
        'task': 'chrominer',
        'compounds': [],
        'mztol': mztol,
        'IsotopesAsSamples': 'true',
        'data': [],
    }
    assert adf.columns.isin(['id', 'formula', 'ion']).sum() == 3

    adf['formion'] = adf.apply(lambda x: (Formula(x['formula']) + parse_ion(x['ion'])).to_chnops(), axis=1)
    def make_idion(row):
        if row['ion'] == '-H(-)':
            return row['id']
        else:
            return f'{row["id"]} | {row["ion"]}'
    adf['idion']   = adf.apply(make_idion, axis=1)
    adf['nC'] = adf['formula'].apply(lambda x: Formula(x)['C'])

    for i, (idion, formion, nC) in adf.loc[:, ['idion', 'formion', 'nC']].iterrows():
        compound = {
            'name': f'{idion}',
            'formula': formion,
            'RT': 0,
            'RTwin': 100, # this is completely uninformative, so that the whole spectrum is considered
        }
        if isinstance(CIso, int):
            compound['CIso'] = CIso
        elif CIso == True:
            compound['CIso'] = nC

        chromin['compounds'].append(compound)

    minsauer = folder.replace(r'\\d.ethz.ch\groups\biol\sysbc\sauer_1', '')
    for file in os.listdir(folder):
        datum = {
            'file': PureWindowsPath(os.path.join(minsauer, file)).as_posix(),
            'label': re.sub('\.d$|\.raw$', repl='', string=file),
        }
        chromin['data'].append(datum)
    # with open('data.trigger', 'w') as f:
    #     f.write(jsonf)
    return json.dumps(chromin, indent=4)


if __name__ == '__main__':
    from sbmfi.settings import BASE_DIR
    import os

    file = os.path.join(BASE_DIR, 'rieke_hplc', 'wtool_input.xlsx')
    print(file)
    # file = "/measurement_model/coli_observation_model_dilutions.xlsx"
    #
    df = pd.read_excel(file)
    # openbis = openbis_registration_file(df=df)
    openbis = openbis_upload(df=df, tsv_file=os.path.join(BASE_DIR, 'rieke_hplc', 'from_openbis_file.tsv'), project_name='rieke_hplc', double_inj=True)
    openbis.to_csv(os.path.join(BASE_DIR, 'rieke_hplc', 'openbis_upload.tsv'), index=False, sep='\t')

    # df = df.drop(['nat_coli_ul', 'C13_coli_ul', 'Comment'], axis=1)
    # wkl = tof6546_worklist(df, project_name='rieke_HPLC', method_map={'A': None})
    # wkl.to_csv(os.path.join(BASE_DIR, 'rieke_hplc', 'wkl_50_rieke_HPLC.csv'), index=False)
    # #wkl.to_csv(os.path.join(BASE_DIR,'wkl_46_meas_mod_LCMS.csv'), index=False)
    # #df.to_csv(os.path.join(BASE_DIR,'INPUT_meas_mod_LCMS.csv'), index=False)
    # print(df)
    # print(wkl)
    #
    # print(wkl['Data File'].unique().shape)
    #


    # folder = r'\\d.ethz.ch\groups\biol\sysbc\sauer_1\users\Tomek\ALAA LC-MS first file\MM_FormicAcid_only_test_NEG'
    # adf = pd.read_excel(r"C:\python_projects\pysumo\mm2020.xlsx")
    # jsonf = chrominer_json(adf=adf, folder=folder, CIso=1)
    # file_fadR = "C:\python_projects\pysumo\TF_screen_rike_worklist_T5.xlsx"
    # file_Cra = "C:\python_projects\pysumo\TF_screen_rike_worklist_T3.xlsx"
    # file_rpoN = "C:\python_projects\pysumo\TF_rieke_own_wtool.xlsx"
    # dfadR = pd.read_excel(file_fadR, sheet_name='Sheet1')
    # dCra = pd.read_excel(file_Cra, sheet_name='Sheet1')
    # drpoN = pd.read_excel(file_rpoN, sheet_name='Sheet1')
    # method_map = {'FIA': r"D:\MassHunter\Data\All\method repository\fia_ctc_neg_AF_double_1700record.m"}
    # name = 'TF_screen_rike_worklist_T5'
    # # print(sequencer_source_file_processer(df).head(20))
    # # wkl_6546 = tof6546_worklist(df, project_name=name, method_map=method_map)
    # # wkl_6546.to_csv(f'C:\python_projects\pysumo\{name} wkl_46.csv', index=False)
    # ofadR = openbis_registration_file(df=dfadR)
    # oCra = openbis_registration_file(df=dCra)
    # orpoN = openbis_registration_file(df=drpoN)
    #
    #
    #
    # obis = pd.concat([orpoN, oCra, ofadR], axis=0)
    # obis.to_csv(f"C:\python_projects\pysumo\TF_screen_rieke openbis_register.tsv", index=False, sep='\t')

    # tsv_file = r"C:\python_projects\pysumo\TF_screen_openbis_sample_register.tsv"
    # df = pd.concat([drpoN, dCra, dfadR])
    # updf = openbis_upload(df=df, tsv_file=tsv_file, project_name='TF_screen rpoN Cra fadR',
    #                       double_inj=True)
    # updf.to_csv(f"C:\python_projects\pysumo\TF_screen openbis_upload.tsv", sep='\t', index=False)
    # updf.to_excel('upload.xlsx', index=False)
    # print(updf.head())

    # method_map = {'FIA+': r'D:\MassHunter\Methods\_FIAplus\20200506_FIAplus_1700_NEG_AO_v2.m'}
    # name = 'SUMO_MM FIA eco_nat_u13c ratios MonMx'
    #
    # file =f"C:\python_projects\pysumo\{name}.xlsx"
    # df = pd.read_excel(file, skiprows=0)
    #
    # # wkl_6546 = tof6546_worklist(df, project_name='SUMO_MM_hi_pH_FIA+_and_5ulFIA', method_map=method_map)
    # # openbis = openbis_registration_file(df=df)
    # # print(wkl_6546)
    # # print(openbis)
    # # wkl_6546.to_csv(f'C:\python_projects\pysumo\{name} wkl_46.csv', index=False)
    # # openbis.to_csv(f"C:\python_projects\pysumo\{name} openbis_register.tsv", index=False, sep='\t')
    #
    # tsv_file = f"C:\python_projects\pysumo\{name} from_openbis.tsv"
    # updf = openbis_upload(df=df, tsv_file=tsv_file, project_name='SUMO_MM FIA eco_nat_u13c ratios MonMx', double_inj=True)
    # updf.to_csv(f"C:\python_projects\pysumo\{name} openbis_upload.tsv", sep='\t', index=False)

