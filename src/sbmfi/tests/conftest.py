import pytest
from sbmfi.core.model import (
    LabellingModel,
    EMU_Model,
    create_full_metabolite_kwargs,
    model_builder_from_dict,
    process_biomass_reaction
)

@pytest.fixture
def basic_reaction_kwargs():
    return {
        'a_in': {
            'atom_map_str': '--> A/ab',  # exchange reaction
            'upper_bound': 100.0,
            'lower_bound': 0.0
        },
        'r1': {
            'atom_map_str': 'A/ab + J --> P/ab',  # has a co-factor
            'upper_bound': 100.0,
            'lower_bound': 0.0
        },
        'r2': {
            'atom_map_str': 'A/ab + B/cd --> Q/acdb',  # default bounds
        },
        'r3': {
            'atom_map_str': 'Q/acdb --> R/cd + S/ba',
            'upper_bound': 100.0,
            'lower_bound': 0.0
        },
        'bm': {
            'atom_map_str': 'biomass --> ',  # biomass reaction
            'reaction_str': '0.3H + 0.6P + 0.5B + 0.1Q --> ∅',
            'upper_bound': 100.0,
            'lower_bound': 0.0
        },
        'rp': {
            'atom_map_str': 'A/ab + B/cd + A/ef --> L/acf',  # pseudo-reaction
            'pseudo': True
        },
    }

@pytest.fixture
def basic_metabolite_kwargs():
    return {
        'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
        'B': {'formula': 'C2H6O', 'compartment': 'c', 'charge': 0},
        'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
        # 'Q': {},  # to test a metabolite without annotations
        'R': {}, # metabolite without formula
        'S': {'formula': 'C2H4O3', 'compartment': 'p', 'charge': -1},
        'J': {'formula': 'C1H1O1', 'compartment': 'c', 'charge': 0}  # cofactor
    }

@pytest.fixture
def labelling_model(basic_reaction_kwargs, basic_metabolite_kwargs):
    cobra_model = model_builder_from_dict(basic_reaction_kwargs, basic_metabolite_kwargs)
    model = LabellingModel(cobra_model)
    model.add_labelling_kwargs(basic_reaction_kwargs, basic_metabolite_kwargs)
    return model