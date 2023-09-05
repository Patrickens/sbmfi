import pandas as pd
from collections import OrderedDict
from sbmfi.core.model import LabellingModel, EMU_Model, RatioEMU_Model
from sbmfi.core.cumodel import CumomerModel
from sbmfi.core.linalg import LinAlg
from sbmfi.settings import MODEL_DIR, SIM_DIR
from sbmfi.lcmsanalysis.util import _strip_bigg_rex
import sys, os
import cobra
from cobra.io import read_sbml_model
from cobra import Reaction, Metabolite, DictList, Model
from pta import ConcentrationsPrior

def e_coli_glc(amino_acids='core', load_conc_prior=False):
    # TODO: maybe add methylglyoxal pathway! https://en.wikipedia.org/wiki/Methylglyoxal_pathway
    # TODO: maybe add urea cycle! https://greek.doctor/second-year/biochemistry-1/lectures/12-urea-cycle/
    # Add important arguments to the metabolites for generating the pysumo model; this depends on the experiment we do!
    # NOTE: tracer paper https://www.sciencedirect.com/science/article/pii/S1096717615000038
    # growth rate of 0.87 with bmid_GAM
    model = _read_model(name=amino_acids)

    gluc_ratio_repo = {
        'PYR|glycolysis': {  # Glycolysis/PPP
            'numerator':   {'PGI': 1.0},  # this is the netflux of the pgi reaction
            'denominator': {'PGI': 1.0, 'PGL': 1.0}
        },

        'PYR|EDD': {  # Pyruvate from ED
            'numerator':   {'EDD': 1.0},
            'denominator': {'EDD': 1.0, 'PYK': 1.0, 'ME1': 1.0, 'ME2': 1.0}
        },

        'PEP|PCK': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator':   {'PPCK': 1.0},
            'denominator': {'PPCK': 1.0, 'ENO': 1.0}
        },

        'PYR|MAE': {  # Pyruvate from malic enzyme
            'numerator':   {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'PYK': 1.0, 'EDD': 1.0}
        },

        'OAA|PPC': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator':   {'PPC': 1.0},
            'denominator': {'PPC': 1.0, 'MDH': 1.0}
        },

        'MAL|MALS': {  # glyoxylate shunt
            'numerator': {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0}
        },
    }
    gluc_labeling = pd.Series(OrderedDict({
        'glc__D_e/000000': 0.2,
        'glc__D_e/111111': 0.8,
        'co2_e/0': 0.989,
        'co2_e/1': 1 - 0.989
    }), name='GnGu')

    uptake_ub = -10.0
    if load_conc_prior:
        uptake_ub = 0.0

    glc_ex = model.reactions.get_by_id('EX_glc__D_e')
    glc_ex.bounds = (-10.0, uptake_ub)
    pts = model.reactions.get_by_id('GLCpts')
    pts.bounds = (0.0, 10.0)
    ac_out = model.reactions.get_by_id('ACt2r')
    ac_out.bounds = (-1000.0, 0.0)
    ac_ex = model.reactions.get_by_id('EX_ac_e')
    ac_ex.bounds = (0.0, 1000.0)
    pps = model.reactions.get_by_id('PPS')
    pps.bounds = (0.0, 0.0)
    fbp = model.reactions.get_by_id('FBP')
    fbp.bounds = (0.0, 0.0)
    mdh = model.reactions.get_by_id('MDH')
    mdh.lower_bound = 0.0

    pgi = model.reactions.get_by_id('PGI')
    # pgi.lower_bound = 0.0

    columns = [
        'glc__D_e/000000', 'glc__D_e/100000', 'glc__D_e/010000', 'glc__D_e/001000', 'glc__D_e/011000',
        'glc__D_e/000100', 'glc__D_e/000010', 'glc__D_e/000001', 'glc__D_e/000111', 'glc__D_e/111111',
        'co2_e/0', 'co2_e/1'
    ]
    index = ['Gn', 'G1', 'G2', 'G23', 'G5', 'G6', 'G456', 'G1GU1:1', 'GnGU1:1']
    substrate_df = pd.DataFrame([
        [0.9391618, 0.0101397, 0.0101397, 0.0101397, 0, 0.0101397, 0.0101397, 0.0101397, 0, 0, 0.989, 0.011],  # Gn
        [0,   1,   0, 0, 0, 0, 0, 0, 0, 0,   0.989, 0.011],  # G1
        [0,   0,   1, 0, 0, 0, 0, 0, 0, 0,   0.989, 0.011],  # G2
        [0,   0,   0, 0, 1, 0, 0, 0, 0, 0,   0.989, 0.011],  # G23
        [0,   0,   0, 0, 0, 0, 1, 0, 0, 0,   0.989, 0.011],  # G5
        [0,   0,   0, 0, 0, 0, 0, 1, 0, 0,   0.989, 0.011],  # G6
        [0,   0,   0, 0, 0, 0, 0, 0, 1, 0,   0.989, 0.011],  # G456
        [0,   0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.989, 0.011],  # G1GU1:1
        [0.5, 0,   0, 0, 0, 0, 0, 0, 0, 0.5, 0.989, 0.011],  # GnUG1:1
        # [0.46874652, 0.005208913, 0.005208913, 0.005208913, 0, 0.005208913, 0.005208913, 0.005208913, 0, 0.5, 0.989, 0.011],  # GnUG1:1
    ], index=index, columns=columns)

    conc_prior = None
    if load_conc_prior:
        conc_prior = ConcentrationsPrior.load('ecoli_M9_glc')
        aero_prior = ConcentrationsPrior.load('M9_aerobic')
        conc_prior.add(aero_prior)

    measured_metabolites = pd.Series({
        'gly_c': 1e4,
        'pyr_c': 1e4,
        'ala__L_c': 1e4,
        'lac__D_c': 1e4,
        'ser__L_c': 1e4,
        'pro__L_c': 1e4,
        'fum_c': 1e6,
        'val__L_c': 1e4,
        'succ_c': 1e6,
        'thr__L_c': 1e4,
        #'leu__L_c': 1e4,
        #'ile__L_c': 1e4,
        'asp__L_c': 5e4,
        'mal__L_c': 1e4,
        'akg_c': 5e4,
        'gln__L_c': 1e4,
        'glu__L_c': 1e4,
        'met__L_c': 1e4,
        'pep_c': 1e4,
        'g3p_c': 1e4,
        'acon_C_c': 1e5,
        'tyr__L_c': 1e4,
        '2pg_c': 1e4,
        'cit_c': 1e4,
        'r5p_c': 5e3,
        '2ddg6p_c': 1e4,
        'glc__D_e': 1e4,
    }, name='total_signal')

    toti = measured_metabolites.copy()
    # NB removed bigg compartments to signal that we cannot distinguish them using LCMS
    toti.index = toti.index.str.replace(_strip_bigg_rex, '')

    kwargs = {
        'cobra_model': model,
        'ratio_repo': gluc_ratio_repo,
        'input_labelling': gluc_labeling,
        'substrate_df': substrate_df,
        'concentrations_prior': conc_prior,
        'measured_metabolites': measured_metabolites.index.values,
        'total_intensities': toti,
    }

    return model, kwargs


def e_coli_succ(amino_acids='core', load_conc_prior=False):
    # growth rate of 0.40 with bmid_GAM
    model = _read_model(name=amino_acids)

    succ_ratio_repo = {
        'PEP|PPS': {  # Pyruvate from ED
            'numerator':   {'PPS': 1.0},
            'denominator': {'PPS': 1.0, 'PPCK': 1.0},
        },
        'FBP|PPP': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator':   {'FBP': 1.0},
            'denominator': {'FBP': 1.0, 'TKT2': 1.0, 'TALA': 1.0},
        },

        'PYR|MAE': {  # Pyruvate from malic enzyme
            'numerator':   {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'EDD': 1.0},
        },
        'MAL|MALS': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator':   {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0},
        },
    }
    succ_labeling = pd.Series(OrderedDict({
        'succ_e/0000': 0.2,
        'succ_e/1111': 0.8,
        'co2_e/0': 0.989,
        'co2_e/1': 1 - 0.989
    }), name='SnSu')

    uptake_ub = -10.0
    if load_conc_prior:
        uptake_ub = 0.0

    succ_ex = model.reactions.get_by_id('EX_succ_e')
    succ_ex.bounds = (-10.0, uptake_ub)
    succ_in = model.reactions.get_by_id('SUCCt2_2')
    succ_in.bounds = (0.0, 10.0)
    succ_out = model.reactions.get_by_id('SUCCt3')
    succ_out.bounds = (0.0, 0.0)
    ac_out = model.reactions.get_by_id('ACt2r')
    ac_out.bounds = (-1000.0, 0.0)
    ac_ex = model.reactions.get_by_id('EX_ac_e')
    ac_ex.bounds = (0.0, 1000.0)
    pyk = model.reactions.get_by_id('PYK')
    pyk.bounds = (0.0, 0.0)
    pfk = model.reactions.get_by_id('PFK')
    pfk.bounds = (0.0, 0.0)

    conc_prior = None
    if load_conc_prior:
        conc_prior = ConcentrationsPrior.load('ecoli_M9_succ')
        aero_prior = ConcentrationsPrior.load('M9_aerobic')
        conc_prior.add(aero_prior)


    kwargs = {
        'ratio_repo': succ_ratio_repo,
        'input_labelling': succ_labeling,
        'concentrations_prior': conc_prior,
    }

    return model, kwargs


def e_coli_pyr(amino_acids='core', load_conc_prior=False):
    # growth rate of 10.3 with bmid_GAM
    model = _read_model(name=amino_acids)

    pyr_ratio_repo = {
        'PEP|PPS': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator': {'PPCK': 1.0},
            'denominator': {'PPCK': 1.0, 'PPS': 1.0}
        },
        'PYR|MAE': {  # Pyruvate from malic enzyme
            'numerator': {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'EDD': 1.0}
        },
        'OAA|PPC': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator': {'PPC': 1.0},
            'denominator': {'PPC': 1.0, 'MDH': 1.0}
        },
        'MAL|MALS': {  # glyoxylate shunt
            'numerator': {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0}
        },
    }
    pyr_labeling = pd.Series(OrderedDict({
        'pyr_e/111': 0.8,
        'pyr_e/000': 0.2,
        'co2_e/0': 0.989,
        'co2_e/1': 1 - 0.989
    }), name='PnPu')

    uptake_ub = -10.0
    if load_conc_prior:
        uptake_ub = 0.0

    pyr_ex = model.reactions.get_by_id('EX_pyr_e')
    pyr_ex.bounds = (-10.0, uptake_ub)
    pyr_in = model.reactions.get_by_id('PYRt2')
    pyr_in.bounds = (0.0, 10.0)
    pyk = model.reactions.get_by_id('PYK')
    pyk.bounds = (0.0, 0.0)
    pfk = model.reactions.get_by_id('PFK')
    pfk.bounds = (0.0, 0.0)
    ac_out = model.reactions.get_by_id('ACt2r')
    ac_out.bounds = (-1000.0, 0.0)
    ac_ex = model.reactions.get_by_id('EX_ac_e')
    ac_ex.bounds = (0.0, 1000.0)

    conc_prior = None
    if load_conc_prior:
        conc_prior = ConcentrationsPrior.load('ecoli_M9_pyr')
        aero_prior = ConcentrationsPrior.load('M9_aerobic')
        conc_prior.add(aero_prior)

    columns = [
        'pyr_e/000', 'pyr_e/100', 'pyr_e/010', 'pyr_e/001', 'pyr_e/110',
        'pyr_e/011', 'pyr_e/111', 'co2_e/0', 'co2_e/1'
    ]
    index = ['PnPU1:1', 'P1', 'P2', 'P3', 'P12', 'P23', ]
    substrate_df = pd.DataFrame([
        [0.5, 0, 0, 0, 0, 0, 0.5, 0.989, 0.011],  # PnPU1:1
        [0,   1, 0, 0, 0, 0, 0,   0.989, 0.011],  # P1
        [0,   0, 1, 0, 0, 0, 0,   0.989, 0.011],  # P2
        [0,   0, 0, 1, 0, 0, 0,   0.989, 0.011],  # P3
        [0,   0, 0, 0, 1, 0, 0,   0.989, 0.011],  # G12
        [0,   0, 0, 0, 0, 1, 0,   0.989, 0.011],  # G23
    ], index=index, columns=columns)

    measured_metabolites = pd.Series({
        'gly_c': None,
        'pyr_c': None,
        'ala__L_c': None,
        'lac__D_c': None,
        'ser__L_c': None,
        'pro__L_c': None,
        'fum_c': None,
        'val__L_c': None,
        'succ_c': None,
        'thr__L_c': None,
        # 'leu__L_c': 1e4,
        # 'ile__L_c': 1e4,
        'asp__L_c': None,
        'mal__L_c': None,
        'akg_c': None,
        'gln__L_c': None,
        'glu__L_c': None,
        'met__L_c': None,
        'pep_c': None,
        'g3p_c': None,
        'acon_C_c': None,
        'tyr__L_c': None,
        '2pg_c': None,
        'cit_c': None,
        'r5p_c': None,
        '2ddg6p_c': None,
        'pyr_e': None,
    }, name='total_signal')

    kwargs = {
        'ratio_repo': pyr_ratio_repo,
        'input_labelling': pyr_labeling,
        'concentrations_prior': conc_prior,
    }

    kwargs = {
        'ratio_repo': pyr_ratio_repo,
        'input_labelling': pyr_labeling,
        'substrate_df': substrate_df,
        'concentrations_prior': conc_prior,
        'measured_metabolites': measured_metabolites.index.values,
    }

    return model, kwargs


def e_coli_glyc(amino_acids='core', load_conc_prior=False):
    model = _read_model(name=amino_acids)

    glyc_ratio_repo = {

    }
    glyc_labeling = pd.Series(OrderedDict({
        'glyc_e/111': 0.8,
        'glyc_e/000': 0.2,
        'co2_e/0': 0.989,
        'co2_e/1': 1 - 0.989
    }), name='GLYnGLYu')

    uptake_ub = -10.0
    if load_conc_prior:
        uptake_ub = 0.0

    glyc_ex = model.reactions.get_by_id('EX_glyc_e')
    glyc_ex.bounds = (-10.0, uptake_ub)
    glyc_in = model.reactions.get_by_id('GLYCt')
    glyc_in.bounds = (0.0, 10.0)
    pyk = model.reactions.get_by_id('PPS')
    pyk.bounds = (0.0, 0.0)
    pfk = model.reactions.get_by_id('PFK')
    pfk.bounds = (0.0, 0.0)

    conc_prior = None
    if load_conc_prior:
        conc_prior = ConcentrationsPrior.load('ecoli_M9_glyc')
        aero_prior = ConcentrationsPrior.load('M9_aerobic')
        conc_prior.add(aero_prior)

    kwargs = {
        'ratio_repo': glyc_ratio_repo,
        'input_labelling': glyc_labeling,
        'concentrations_prior': conc_prior,
    }

    return model, kwargs


def e_coli_xyl(amino_acids='core', load_conc_prior=False):
    # growth rate of 0.71 with bmid_GAM
    model = _read_model(name=amino_acids)

    xyl_ratio_repo = {
        'PYR|EDD': {  # Pyruvate from ED
            'numerator': {'EDD': 1.0},
            'denominator': {'EDD': 1.0, 'PYK': 1.0, 'ME1': 1.0, 'ME2': 1.0}
        },

        'PEP|PCK': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator': {'PPCK': 1.0},
            'denominator': {'PPCK': 1.0, 'ENO': 1.0}
        },

        'PYR|MAE': {  # Pyruvate from malic enzyme
            'numerator': {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'PYK': 1.0, 'EDD': 1.0}
        },

        'OAA|PPC': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator': {'PPC': 1.0},
            'denominator': {'PPC': 1.0, 'MDH': 1.0}
        },

        'MAL|MALS': {  # glyoxylate shunt
            'numerator': {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0}
        },
    }
    xyl_labeling = pd.Series(OrderedDict({
        'xyl__D_e/11111': 0.8,
        'xyl__D_e/00000': 0.2,
        'co2_e/0': 0.989,
        'co2_e/1': 1 - 0.989
    }), name='XnXu')

    uptake_ub = -10.0
    if load_conc_prior:
        uptake_ub = 0.0

    xyl_ex = model.reactions.get_by_id('EX_xyl__D_e')
    xyl_ex.bounds = (-10.0, uptake_ub)
    xyl_in1 = model.reactions.get_by_id('XYLt2')
    xyl_in1.bounds = (0.0, 10.0)
    xyl_in2 = model.reactions.get_by_id('XYLabc')
    xyl_in2.bounds = (0.0, 10.0)
    ac_out = model.reactions.get_by_id('ACt2r')
    ac_out.bounds = (-1000.0, 0.0)
    ac_ex = model.reactions.get_by_id('EX_ac_e')
    ac_ex.bounds = (0.0, 1000.0)
    pps = model.reactions.get_by_id('PPS')
    pps.bounds = (0.0, 0.0)
    pfk = model.reactions.get_by_id('PFK')
    pfk.bounds = (0.0, 0.0)

    conc_prior = None

    columns = [
        'xyl__D_e/00000', 'xyl__D_e/10000', 'xyl__D_e/11111', 'co2_e/0', 'co2_e/1'
    ]
    index = ['XnXU1:1', 'X1']
    substrate_df = pd.DataFrame([
        [0.5, 0, 0.5, 0.989, 0.011],  # XnXU1:1
        [0,   1, 0,   0.989, 0.011],  # X1
    ], index=index, columns=columns)


    measured_metabolites = pd.Series({
        'gly_c': None,
        'pyr_c': None,
        'ala__L_c': None,
        'lac__D_c': None,
        'ser__L_c': None,
        'pro__L_c': None,
        'fum_c': None,
        'val__L_c': None,
        'succ_c': None,
        'thr__L_c': None,
        # 'leu__L_c': 1e4,
        # 'ile__L_c': 1e4,
        'asp__L_c': None,
        'mal__L_c': None,
        'akg_c': None,
        'gln__L_c': None,
        'glu__L_c': None,
        'met__L_c': None,
        'pep_c': None,
        'g3p_c': None,
        'acon_C_c': None,
        'tyr__L_c': None,
        '2pg_c': None,
        'cit_c': None,
        'r5p_c': None,
        '2ddg6p_c': None,
        'xyl__D_e': None,
    }, name='total_signal')

    kwargs = {
        'ratio_repo': xyl_ratio_repo,
        'input_labelling': xyl_labeling,
        'substrate_df': substrate_df,
        'concentrations_prior': conc_prior,
        'measured_metabolites': measured_metabolites.index.values,
    }

    return model, kwargs


def e_coli_pyr_xyl(amino_acids='core', load_conc_prior=False):
    model = _read_model(name=amino_acids)
    xyl_ratio_repo = {
        'PYR|EDD': {  # Pyruvate from ED
            'numerator': {'EDD': 1.0},
            'denominator': {'EDD': 1.0, 'PYK': 1.0, 'ME1': 1.0, 'ME2': 1.0}
        },

        'PEP|PCK': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator': {'PPCK': 1.0},
            'denominator': {'PPCK': 1.0, 'ENO': 1.0}
        },

        'PYR|MAE': {  # Pyruvate from malic enzyme
            'numerator': {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'PYK': 1.0, 'EDD': 1.0}
        },

        'OAA|PPC': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator': {'PPC': 1.0},
            'denominator': {'PPC': 1.0, 'MDH': 1.0}
        },

        'MAL|MALS': {  # glyoxylate shunt
            'numerator': {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0}
        },
    }

    columns = [
        'pyr_e/000', 'pyr_e/100', 'pyr_e/010', 'pyr_e/001', 'pyr_e/110',
        'pyr_e/011', 'pyr_e/111',
        'xyl__D_e/00000', 'xyl__D_e/10000', 'xyl__D_e/11111',
        'co2_e/0', 'co2_e/1'
    ]
    index = ['PnXU1:1', 'P1', 'P2', 'P3', 'P12', 'P23', ]
    substrate_df = pd.DataFrame([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.989, 0.011],  # PnXU1:1
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.989, 0.011],  # PUXn1:1
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0.989, 0.011],  # PUX11:1
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.989, 0.011],  # P3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.989, 0.011],  # G12
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.989, 0.011],  # G23
    ], index=index, columns=columns)


if __name__ == "__main__":
   pass