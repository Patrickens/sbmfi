import pandas as pd
from collections import OrderedDict
from sbmfi.core.model import LabellingModel, EMU_Model, RatioEMUModel
from sbmfi.inference.simulator import DataSetSim
from sbmfi.core.observation import ClassicalObservationModel, LCMS_ObservationModel, MVN_BoundaryObservationModel
from sbmfi.core.linalg import LinAlg
from sbmfi.models.build_models import simulator_factory
from sbmfi.settings import MODEL_DIR, SIM_DIR
from sbmfi.lcmsanalysis.util import _strip_bigg_rex
import sys, os
import cobra
from cobra.io import read_sbml_model
from cobra import Reaction, Metabolite, DictList, Model


def spiro(
        backend='numpy', auto_diff=False, batch_size=1, add_biomass=True, v2_reversible=False,
        ratios=True, build_simulator=False, add_cofactors=True, which_measurements=None, seed=2,
        which_labellings=None, logit_xch_fluxes=False, include_bom=True,
):
    # NOTE: this one has 2 interesting flux ratios!
    # NOTE this has been parametrized to exactly match the Wiechert fml file: C:\python_projects\pysumo\src\sumoflux\models\fml\spiro.fml
    #   which is a slightly modified version of the one found on: https://github.com/modsim/FluxML/blob/master/examples/models/spirallus_model_level_1.fml
    # for m in M.metabolites: # NOTE: needed to store as sbml
    #     m.compartment = 'c'
    # sbml_file = os.path.join(MODEL_DIR, 'sbml', 'spiro.xml')
    # json_file = os.path.join(MODEL_DIR, 'escher_input', 'model', 'spiro.json')
    # cobra.io.write_sbml_model(cobra_model=M, filename=sbml_file)
    # cobra.io.save_json_model(model=M, filename=json_file)
    if (which_measurements is not None) and not build_simulator:
        raise ValueError

    reaction_kwargs = {
        'a_in': {
            'lower_bound': 10.0, 'upper_bound': 10.0,
            'atom_map_str': '∅ --> A/ab'
        },
        # 'a_in': {
        #     'lower_bound': -10.0, 'upper_bound': -10.0,
        #     'atom_map_str': 'A/ab --> ∅'
        # },
        'd_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/abc --> ∅'
        },
        'f_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'F/a --> ∅'
        },
        'h_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'H/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab --> B/ab'
        },
        'v2': {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            'rho_min': 0.1, 'rho_max': 0.8,
            'atom_map_str': 'B/ab ==> E/ab'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + E/cd --> C/abcd'
        },
        'v4': {
            'upper_bound': 100.0, # 'lower_bound': -10.0,
            'atom_map_str': 'E/ab --> H/ab'
        },
        # 'v5': {
        #     'upper_bound': 100.0,
        #     'atom_map_str': 'C/abcd --> F/a + D/bcd'
        # },
        'v5': {  # NB this is an always reverse reaction!
            'lower_bound': -100.0, # 'upper_bound': 100.0
            'atom_map_str': 'F/a + D/bcd  <-- C/abcd',  # <--  ==>
            # 'atom_map_str': 'F/a + D/bcd  <=> C/abcd',  # <--  ==>
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/abc --> E/ab + F/c'
        },
        'v7': {
            'upper_bound': 100.0,
            'atom_map_str': 'F/a + F/b --> H/ab'
        },
        'vp': {
            'lower_bound': 0.0, # 'upper_bound': 100.0,
            'pseudo': True,
            'atom_map_str': 'C/abcd + D/efg + H/hi --> L/abgih'
        },
    }
    metabolite_kwargs = {
        'A': {'formula': 'C2H4O5'},
        'B': {'formula': 'C2HPO3'},
        'C': {'formula': 'C4H6N4OS'},
        'D': {'formula': 'C3H2'},
        'E': {'formula': 'C2H4O5'},
        'F': {'formula': 'CH2'},
        'G': {'formula': 'CH2'},  # not used
        'H': {'formula': 'C2H2'},
        'L': {'formula': 'C5KNaSH'},  # pseudo-metabolite
        'P': {'formula': 'C2H'},
    }

    substrate_df = pd.DataFrame([
        [0.2, 0.0, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
    ], columns=['A/00', 'A/01', 'A/10', 'A/11'], index=list('ABCDE'))

    if which_labellings is not None:
        substrate_df = substrate_df.loc[which_labellings]

    ratio_repo = {
        'E|v2': {
            'numerator': OrderedDict({'v2': 1}),
            'denominator': OrderedDict({'v2': 1, 'v6': 1})},
        'H|v4': {
            'numerator': OrderedDict({'v4': 1}),
            'denominator': OrderedDict({'v7': 1, 'v4': 1})},
            # 'denominator': OrderedDict({'v6': 1, 'v4': 1})},  # make ratios correlated
    }

    if not v2_reversible:
        reaction_kwargs['v2'] = {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            'atom_map_str': 'B/ab --> E/ab'
        }
        ratio_repo['E|v2']= {
            'numerator': OrderedDict({'v2': 1}),
            'denominator': OrderedDict({'v2': 1, 'v6': 1})
        }

    annotation_df = pd.DataFrame([
        ('H', 0, 'M-H', 2.0, 0.01, 3e3),
        ('H', 1, 'M-H', 3.0, 0.01, 3e3),

        ('H', 1, 'M+F', 5.0, 0.03, 2e3),
        ('H', 1, 'M+Cl', 88.0, 0.03, 2e3),
        ('H', 0, 'M+F', 4.0, 0.03, 2e3),  # NOTE: to indicate that da_df is not yet in any order!
        ('H', 0, 'M+Cl', 89.0, 0.03, 2e3),

        # ('B', 0, 'M-H', 6.0, 0.01),
        # ('B', 1, 'M-H', 7.0, 0.01),

        ('P', 1, 'M-H', 3.7, 0.01),  # NOTE: an annotated metabolite that is not in the model
        ('P', 2, 'M-H', 4.7, 0.01),
        ('P', 3, 'M-H', 5.7, 0.01),

        ('C', 0, 'M-H', 1.5, 0.02, 7e3),
        ('C', 3, 'M-H', 4.5, 0.02, 7e3),
        ('C', 4, 'M-H', 5.5, 0.02, 7e3),

        ('D', 2, 'M-H', 12.0, 0.01, 1e5),
        ('D', 0, 'M-H', 9.0, 0.01, 1e5),
        ('D', 3, 'M-H', 13.0, 0.01, 1e5),

        ('L|[1,2]', 0, 'M-H', 14.0, 0.01, 4e4),
        ('L|[1,2]', 1, 'M-H', 15.0, 0.01, 4e4),

        ('L', 0, 'M-H', 14.0, 0.01, 4e5),
        ('L', 1, 'M-H', 15.0, 0.01, 4e5),
        ('L', 2, 'M-H', 16.0, 0.01, 4e5),
        ('L', 5, 'M-H', 19.0, 0.01, 4e5),
    ], columns=['met_id', 'nC13', 'adduct_name', 'mz', 'sigma_x', 'total_I'])
    formap = {k: v['formula'] for k, v in metabolite_kwargs.items()}
    annotation_df['formula'] = annotation_df['met_id'].map(formap)

    model = simulator_factory(
        id_or_file_or_model='spiro',
        backend=backend,
        auto_diff=auto_diff,
        metabolite_kwargs=metabolite_kwargs,
        reaction_kwargs=reaction_kwargs,
        input_labelling=substrate_df.loc['C'],
        ratio_repo=ratio_repo,
        measurements=annotation_df['met_id'].unique(),
        batch_size=batch_size,
        ratios=ratios,
        build_simulator=build_simulator,
        seed=seed,
        logit_xch_fluxes=logit_xch_fluxes,
    )

    if add_biomass:
        bm = Reaction(id='bm', lower_bound=0.05, upper_bound=1.5)
        bm.add_metabolites(metabolites_to_add={
            model.metabolites.get_by_id('H'): -0.3,
            model.metabolites.get_by_id('B'): -0.6,
            model.metabolites.get_by_id('E'): -0.5,
            model.metabolites.get_by_id('C'): -0.1,
        })
        model.add_reactions(reaction_list=[bm], reaction_kwargs={bm.id: {'atom_map_str': 'biomass --> ∅'}})

    fluxes = {
        'a_in':     1.0,
        'd_out':    0.1,
        'f_out':    0.1,
        'h_out':    0.8,
        'v1':       1.0,
        'v2':       0.7,
        'v2_rev':   0.35,
        'v3':       0.7,
        'v4':       0.2,
        'v5':       0.3,
        'v5_rev':   1.0,
        # 'v5_rev':   0.7,
        'v6':       0.6,
        'v7':       0.6,
    }
    if add_biomass:
        fluxes = {
            'a_in':   10.00,
            # 'a_in_rev':   10.00,
            'd_out':  0.00,
            'f_out':  0.00,
            'h_out':  7.60,
            'v1':     10.00,
            'v2':     1.80,
            'v2_rev': 0.90,
            'v3':     8.20,
            'v4':     0.00,
            'v5':     0.05,
            'v5_rev': 8.10,
            # 'v5_rev': 8.05,
            'v6':     8.05,
            'v7':     8.05,
            'bm':     1.50,
        }
        bm = model.reactions.get_by_id('bm')
        model.objective = {bm: 1}
    else:
        model.objective = {bm: 1}

    if not v2_reversible:
        fluxes['v2'] = fluxes['v2'] - fluxes.pop('v2_rev')

    if add_cofactors:
        cof = Metabolite('cof', formula='H2O')
        v3 = model.reactions.get_by_id('v3')
        v3.add_metabolites({cof: 1})
        ex_cof = Reaction('EX_cof', lower_bound=0.0, upper_bound=1000.0)
        ex_cof.add_metabolites({cof: -1})
        model.add_reactions([ex_cof])
        fluxes['EX_cof'] = fluxes['v3']

    fluxes = pd.Series(fluxes, name='v')
    if (batch_size == 1) and build_simulator:
        model.set_fluxes(fluxes=fluxes)

    biomass_id = 'bm' if add_biomass else None
    measured_boundary_fluxes = ['d_out', 'h_out']
    if add_biomass:
        measured_boundary_fluxes.append(biomass_id)

    measurements, datasetsim, theta = None, None, None
    if which_measurements is not None:
        observation_df = LCMS_ObservationModel.generate_observation_df(model, annotation_df)

        if which_measurements == 'lcms':
            annotation_dfs = {labelling_id: annotation_df for labelling_id in substrate_df.index}
            total_intensities = {}
            unique_ion_ids = observation_df.drop_duplicates(subset=['ion_id'])
            for _, row in unique_ion_ids.iterrows():
                total_intensities[row['ion_id']] = annotation_df.loc[
                    (annotation_df['met_id'] == row['met_id']) & (annotation_df['adduct_name'] == row['adduct_name']),
                    'total_I'
                ].values[0]
            total_intensities = pd.Series(total_intensities)
            obsmods = LCMS_ObservationModel.build_models(model, annotation_dfs, total_intensities=total_intensities)
        elif which_measurements == 'com':
            sigma_ii = {}
            for mid, row in observation_df.iterrows():
                sigma_ii[mid] = annotation_df.loc[
                    (annotation_df['met_id'] == row['met_id']) & (annotation_df['adduct_name'] == row['adduct_name']),
                    'sigma_x'
                ].values[0]
            sigma_ii = pd.Series(sigma_ii)
            annotation_dfs = {labelling_id: (annotation_df, sigma_ii) for labelling_id in substrate_df.index}
            obsmods = ClassicalObservationModel.build_models(model, annotation_dfs)
        else:
            raise ValueError
        bom = None
        if include_bom:
            bom = MVN_BoundaryObservationModel(model, measured_boundary_fluxes, biomass_id)
        datasetsim = DataSetSim(model, substrate_df, obsmods, bom)
        theta = model._fcm.map_fluxes_2_theta(fluxes.to_frame().T, pandalize=True)
        datasetsim.set_true_theta(theta.iloc[0])
        measurements = datasetsim.simulate_true_data(n_obs=1)

    kwargs = {
        'annotation_df': annotation_df,
        'substrate_df': substrate_df,
        'measured_boundary_fluxes': measured_boundary_fluxes,
        'measurements': measurements,
        'fluxes': fluxes,
        'theta': theta,
        'datasetsim': datasetsim,
    }

    return model, kwargs


def heteroskedastic(
        backend='numpy', auto_diff=False, return_type='mdv', batch_size=1, build_simulator=True
):
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound' : 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + co2/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + co2/a'
        },
        'v3': { # TODO: maybe make this one reversible to display thermodynamic effects
            'upper_bound': 100.0,'lower_bound': 4.0, # 'rho_min':0.0, 'rho_max':0.95,
            'atom_map_str': 'B/ab --> D/ab'
        },
        'v4': {
            'upper_bound': 100.0,'lower_bound': 2.0,
            'atom_map_str': 'C/ab --> D/ab'
        },
        'v5': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/ab --> E/ab',
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab --> E/ab'
        },
    }
    input_labelling = OrderedDict({
        'A/110': 1.0,
    })
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
        'θ2': {
            'numerator': OrderedDict({'v5': 1}),
            'denominator': OrderedDict({'v5': 1, 'v6': 1})
        },
    }
    measurements = ['E'] # , 'D'
    model = simulator_factory(
        id_or_file_or_model='heteroskedastic',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
        build_simulator=build_simulator,
    )
    e_out = model.reactions.get_by_id('e_out')
    model.objective = {e_out: 1}
    return model


def multi_modal(
        backend='numpy',
        auto_diff=False,
        batch_size=1,
        ratios=True,
        prepend_input=False,
):
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound': 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + D/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + D/a'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + D/c --> E/ac + co2/b'
        },
        'v4': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab + D/c --> E/cb + co2/a'
        },
    }
    metabolite_kwargs = {
        'E': {'formula': 'C2H4O2'},
    }
    input_met = 'A'
    if prepend_input:
        tol = 0.01
        reaction_kwargs['I_in'] = {
            'upper_bound': 10 + tol, 'lower_bound': 10.0 - tol,
            'atom_map_str': '∅ --> I/abc'
        }
        reaction_kwargs['a_in'] = {
            'upper_bound': 10 + tol, 'lower_bound': 10.0 - tol,
            'atom_map_str': 'I/abc --> A/abc'
        }
        input_met = 'I'
    input_labelling = pd.Series(OrderedDict({
        f'{input_met}/011': 1.0,
    }), name='input')

    ratio_repo = {
        'r1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
    }

    measurements=['E']
    m = simulator_factory(
        id_or_file_or_model='multi_modal',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        metabolite_kwargs=metabolite_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
        ratios=ratios,
    )
    m.set_input_labelling(input_labelling=input_labelling)
    sdf = m.input_labelling.to_frame().T

    annotation_df = pd.DataFrame([
        ['E', 'C2H4O2', 'M-H', 0],
        ['E', 'C2H4O2', 'M-H', 1],
        ['E', 'C2H4O2', 'M-H', 2],
    ], columns=['met_id', 'formula', 'adduct_name', 'nC13'])
    hdfile = os.path.join(SIM_DIR, 'multi_modal.h5')
    kwargs = {
        'substrate_df': sdf,
        'hdfile': hdfile,
        'annotation_df': annotation_df,
        'total_intensities': pd.Series([1e4], index=['E']),
    }
    return m, kwargs

def polytope_volume(algorithm='emu', backend='numpy', auto_diff=False, return_type='mdv', batch_size=1):
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> E/ab + co2/c'
        },
        # 'v2': {
        #     'upper_bound': 100.0,
        #     'atom_map_str': 'A/abc --> E/ac + co2/b'
        # },
        'v2': {
            'upper_bound': 0.0, 'lower_bound': -100.0,
            'atom_map_str': 'E/ac + co2/b --> A/abc'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> E/bc + co2/a'
        },
    }
    input_labelling = pd.Series(OrderedDict({
        'A/011': 1.0,
    }), name='l')
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v1': 1}),
            'denominator': OrderedDict({'v1': 1, 'v2': 1, 'v3': 1})
        },
    }

    measurements=['E']
    model = simulator_factory(
        id_or_file_or_model='polytope_volume',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
        build_simulator=False,
    )
    model.set_input_labelling(input_labelling=input_labelling)
    model.objective = {model.reactions[4]: 1}
    return model


def thermodynamic(algorithm='emu', backend='numpy', auto_diff=False, return_type='mdv', batch_size=1, v5_reversible=False):
    # TODO: should demonstrate why including thermodynamics matters
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound': 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'd_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'D/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + co2/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + co2/a'
        },
        'v3': {
            'upper_bound': 100.0, 'lower_bound': -100.0,  # 'rho_min':0.0, 'rho_max':0.95,
            'dgibbsr': -6.0, '_sigma_dgibbsr': 4.0,
            'atom_map_str': 'B/ab <=> D/ab'
        },
        'v4': {
            'upper_bound': 100.0, 'lower_bound': 0.0,
            'atom_map_str': 'C/ab --> D/ab'
        },
        'v5': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab --> E/ab',
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab --> E/ab'
        },
    }

    input_labelling = pd.Series(OrderedDict({
        'A/011': 1.0,
    }), name='l')
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
        'θ2': {
            'numerator': OrderedDict({'v5': 1}),
            'denominator': OrderedDict({'v5': 1, 'v6': 1})
        },
    }

    if v5_reversible:
        reaction_kwargs['v5'] = {  # this would yield crazy stuff!
            'upper_bound': 100.0, 'lower_bound': -100.0,
            'dgibbsr': -10.0, '_sigma_dGr': 4.0,
            'atom_map_str': 'B/ab --> E/ab',
        }
        ratio_repo['θ2'] = {
            'numerator': OrderedDict({'v5': 1, 'v5_rev': -1}),
            'denominator': OrderedDict({'v5': 1, 'v5_rev': -1, 'v6': 1})
        }


    measurements=['E']
    M = simulator_factory(
        id_or_file_or_model='thermodynamic',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
    )
    M.set_input_labelling(input_labelling=input_labelling)
    return M


if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    m, k = spiro(backend='torch', add_biomass=False,
        batch_size=1, which_measurements='lcms', build_simulator=True, which_labellings=list('CD'), v2_reversible=True
    )