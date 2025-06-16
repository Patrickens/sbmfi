import pytest
import numpy as np
from cobra import Reaction, Metabolite, Model
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU
from sbmfi.core.model import LabellingModel, EMU_Model

@pytest.fixture
def metabolite_kwargs():
    return {
        # Regular metabolites with different formulas
        'A': {'formula': 'C2H4O2', 'symmetric': False},  # Acetate
        'B': {'formula': 'C2H6O', 'symmetric': False},   # Ethanol
        'P': {'formula': 'C2H3O2', 'symmetric': False, 'compartment': '_c', 'charge': -1},  # Pyruvate
        'Q': {'formula': 'C4H6O4', 'symmetric': False, 'compartment': '_c'},  # Succinate
        'R': {'formula': 'C2H5O2', 'symmetric': False},  # Glycolate
        'S': {'formula': 'C2H4O3', 'symmetric': False},  # Glyoxylate
        'T': {'formula': 'C2H3O3', 'symmetric': False},  # Oxaloacetate
        'U': {'formula': 'C2H4O4', 'symmetric': False},  # Oxalate
        
        # Symmetric metabolites
        'SP': {'formula': 'C2H3O2', 'symmetric': True},  # Pyruvate
        'SQ': {'formula': 'C4H6O4', 'symmetric': True},  # Succinate
        'SR': {'formula': 'C2H5O2', 'symmetric': True},  # Glycolate
        'SS': {'formula': 'C2H4O3', 'symmetric': True},  # Glyoxylate
        'ST': {'formula': 'C2H3O3', 'symmetric': True},  # Oxaloacetate
        'SU': {'formula': 'C2H4O4', 'symmetric': True},  # Oxalate
        
        # Edge cases
        'E1': {'formula': 'C1H4O', 'symmetric': False},  # Single carbon
        'E2': {'formula': 'C6H12O6', 'symmetric': False}, # Large molecule
        'E3': {'formula': 'C0H2O', 'symmetric': False},  # No carbon
        
        # Pseudo metabolites
        'L': {'formula': 'C5H8O2'},  # Pseudo metabolite for testing
        'M': {'formula': 'C3H6O'},   # Another pseudo metabolite
    }

@pytest.fixture
def reaction_kwargs():
    return {
        # Original reactions
        'r1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab --> P/ab'
        },
        'r2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab + B/cd --> Q/acdb'
        },
        'r3': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/acdb --> R/cd + S/ba'
        },
        'r4': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab + B/cd --> T/ac + U/db'
        },
        
        # Symmetric reactions
        'sr1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab --> SP/ab'
        },
        'ssr1': {
            'upper_bound': 100.0,  # this is to test _rev_reaction is symmetric and that forward reaction is same as 'r1'
            'atom_map_str': 'SA/ab --> P/ab'
        },
        'sr2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab + B/cd --> SQ/acdb'
        },
        'sr3': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/acdb --> SR/cd + SS/ba'
        },
        'sr4': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab + B/cd --> ST/ac + SU/db'
        },
        
        # Edge case reactions
        'er1': {
            'upper_bound': 100.0,  # should error due to unbalanced number of carbons
            'atom_map_str': 'E1/a --> E2/abcdef'  # Single to multiple carbons
        },
        'er2': {
            'upper_bound': 100.0, # should error due to unbalanced number of carbons
            'atom_map_str': 'E2/abcdef --> E1/a + E1/b'  # Multiple to single carbons
        },
        
        # Pseudo reactions
        'pr1': {
            'upper_bound': 100.0,
            'pseudo': True,
            'atom_map_str': 'A/ab + B/cd --> L/abcd'  # Simple combination
        },
        'pr3': {
            'upper_bound': 100.0,
            'pseudo': True,
            'atom_map_str': 'A/ab + B/cd + P/fg --> L/acfd'  # unbalanced carbons, should not error
        }
    }


