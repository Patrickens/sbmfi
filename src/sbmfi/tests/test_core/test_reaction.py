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

class TestLabellingReactionCreation:
    """Tests for creating LabellingReaction instances without a model"""
    
    @pytest.fixture
    def basic_reaction(self):
        """Create a basic cobra Reaction without a model"""
        rxn = Reaction('test_rxn')
        met_a = Metabolite('A', formula='C2H4O2')
        met_p = Metabolite('P', formula='C2H4O2')
        rxn.add_metabolites({
            met_a: -1,
            met_p: 1
        })
        return rxn

    @pytest.fixture
    def basic_labelled_reaction(self, basic_reaction):
        """Create a basic LabellingReaction from the basic_reaction"""
        return LabellingReaction(basic_reaction)

    @pytest.fixture
    def pseudo_reaction(self):
        """Create a cobra Reaction for pseudo reaction testing"""
        rxn = Reaction('pseudo_rxn')
        met_a = Metabolite('A', formula='C2H4O2')
        met_b = Metabolite('B', formula='C2H4O2')
        met_l = Metabolite('L', formula='C3H6O3')
        rxn.add_metabolites({
            met_a: -1,
            met_b: -1,
            met_l: 1
        })
        return rxn

    @pytest.fixture
    def pseudo_labelled_reaction(self, pseudo_reaction):
        """Create a LabellingReaction with pseudo=True"""
        return LabellingReaction(pseudo_reaction, pseudo=True)

    @pytest.mark.parametrize("reaction,expected", [
        ("basic_reaction", {
            "id": "test_rxn",
            "num_metabolites": 2,
            "coefficients": [-1, 1],
            "is_labelled": False,
            "is_pseudo": False
        }),
        ("basic_labelled_reaction", {
            "id": "test_rxn",
            "num_metabolites": 2,
            "coefficients": [-1, 1],
            "is_labelled": True,
            "is_pseudo": False
        }),
        ("pseudo_labelled_reaction", {
            "id": "pseudo_rxn",
            "num_metabolites": 3,
            "coefficients": [-1, -1, 1],
            "is_labelled": True,
            "is_pseudo": True
        })
    ])
    def test_reaction_creation(self, request, reaction, expected):
        """Test creation of different types of reactions"""
        # Get the reaction fixture by name
        rxn = request.getfixturevalue(reaction)
        
        # Test basic properties
        assert rxn.id == expected["id"]
        assert len(rxn.metabolites) == expected["num_metabolites"]
        
        # Test metabolite coefficients
        metabolites = list(rxn.metabolites.items())
        assert len(metabolites) == expected["num_metabolites"]
        for i, coef in enumerate(expected["coefficients"]):
            assert metabolites[i][1] == coef
        
        # Test LabellingReaction specific properties
        if expected["is_labelled"]:
            assert isinstance(rxn, LabellingReaction)
            assert rxn.pseudo == expected["is_pseudo"]

    def test_invalid_reaction_creation(self):
        """Test that creating a LabellingReaction from another LabellingReaction raises error"""
        rxn = Reaction('test')
        labelled_rxn = LabellingReaction(rxn)
        with pytest.raises(NotImplementedError):
            LabellingReaction(labelled_rxn)

    def test_reaction_bounds(self, basic_labelled_reaction):
        """Test that reaction bounds are properly set and accessed"""
        basic_labelled_reaction.lower_bound = -10.0
        basic_labelled_reaction.upper_bound = 20.0
        assert basic_labelled_reaction.lower_bound == -10.0
        assert basic_labelled_reaction.upper_bound == 20.0
        assert basic_labelled_reaction.bounds == (-10.0, 20.0)

    def test_reaction_metabolite_manipulation(self):
        """Test that metabolite manipulation after creation is not allowed"""
        rxn = Reaction('test')
        met = Metabolite('X', formula='C1H2O')
        rxn.add_metabolites({met: 1})
        labelled_rxn = LabellingReaction(rxn)
        
        # Attempting to add metabolites should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            labelled_rxn.add_metabolites({Metabolite('Y', formula='C1H2O'): 1})
        
        # Attempting to subtract metabolites should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            labelled_rxn.subtract_metabolites({met: 1})


