import pytest
import numpy as np
from cobra import Reaction, Metabolite, Model
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU
from sbmfi.core.model import LabellingModel, EMU_Model, model_builder_from_dict
from sbmfi.core.linalg import LinAlg

@pytest.fixture
def metabolite_kwargs():
    return {
        # Regular metabolites with different formulas
        'A': {'formula': 'C2H4O2', 'symmetric': False},  # Acetate
        'B': {'formula': 'C2H6O', 'symmetric': False},   # Ethanol
        'P': {'formula': 'C2H3O2', 'symmetric': False, 'compartment': 'c', 'charge': -1},  # Pyruvate
        'Q': {'formula': 'C4H6O4', 'symmetric': False, 'compartment': 'c'},  # Succinate
        'R': {'formula': 'C2H5O2', 'symmetric': False},  # Glycolate
        'S': {'formula': 'C2H4O3', 'symmetric': False},  # Glyoxylate
        'T': {'formula': 'C2H3O3', 'symmetric': False},  # Oxaloacetate
        'U': {'formula': 'C2H4O4', 'symmetric': False},  # Oxalate
        
        # Symmetric metabolites
        'SP': {'formula': 'C2H3O2', 'symmetric': True, 'compartment': 'c', 'charge': -1},  # Pyruvate
        'SQ': {'formula': 'C4H6O4', 'symmetric': True, 'compartment': 'c'},  # Succinate
        'SR': {'formula': 'C2H5O2', 'symmetric': True},  # Glycolate
        'SS': {'formula': 'C2H4O3', 'symmetric': True},  # Glyoxylate
        'ST': {'formula': 'C2H3O3', 'symmetric': True},  # Oxaloacetate
        'SU': {'formula': 'C2H4O4', 'symmetric': True},  # Oxalate
        
        # Edge cases
        'E1': {'formula': 'C1H4O', 'symmetric': False},  # Single carbon
        'E2': {'formula': 'C6H12O6', 'symmetric': False}, # Large molecule
        'E3': {'formula': 'C0H2O', 'symmetric': False},  # No carbon
        
        # Pseudo metabolites
        'L': {'formula': 'C4H8O2'},  # Pseudo metabolite for testing
        'M': {'formula': 'C2H6O'},   # Another pseudo metabolite
    }


@pytest.fixture
def reaction_kwargs():
    return {
        # Original reactions
        'r1': {
            'atom_map_str': 'A/ab --> P/ab'
        },
        'r2': {
            'atom_map_str': 'A/ab + B/cd --> Q/acdb'
        },
        'r3': {
            'atom_map_str': 'A/ab + A/cd --> Q/acdb'
        },
        'r4': {
            'atom_map_str': 'Q/acdb --> R/cd + S/ba'
        },
        'r5': {
            'atom_map_str': 'Q/acdb --> R/cd + R/ba'
        },
        'r6': {
            'atom_map_str': 'A/ab + B/cd --> T/ac + U/db'
        },

        # Symmetric reactions
        'sr1': {
            'atom_map_str': 'A/ab --> SP/ab'
        },
        'sr2': { # this is to test _rev_reaction is symmetric and that forward reaction is same as 'r1'
            'atom_map_str': 'SA/ab <=> P/ab'
        },
        'sr3': {
            'atom_map_str': 'A/ab + B/cd --> SQ/acdb'
        },
        'sr4': {
            'atom_map_str': 'Q/acdb --> SR/cd + S/ba'
        },
        'sr5': {
            'atom_map_str': 'Q/acdb --> SR/cd + SS/ba'
        },
        'sr6': {
            'atom_map_str': 'Q/acdb --> SR/cd + SR/ba'
        },

        # Edge case reactions
        'er1': { # should error due to unbalanced number of carbons
            'atom_map_str': 'E1/a --> E2/abcdef'  # Single to multiple carbons
        },
        'er2': { # should error due to unbalanced number of carbons
            'atom_map_str': 'E2/abcdef --> E1/a + E1/b'  # Multiple to single carbons
        },
        
        # Pseudo reaction
        'pr1': {
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
        ("basic_labelled_reaction", {
            "id": "test_rxn",
            "num_metabolites": 2,
            "coefficients": [-1, 1],
            "is_labelled": True,
            "is_pseudo": False,
            "bounds": (0.0, 1000.0),  # default cobra bounds
            "rho_bounds": (0.0, 0.0),
        }),
        ("pseudo_labelled_reaction", {
            "id": "pseudo_rxn",
            "num_metabolites": 3,
            "coefficients": [-1, -1, 1],
            "is_labelled": True,
            "is_pseudo": True,
            "bounds": (0.0, 0.0),
            "rho_bounds": (0.0, 0.0),
        })
    ])
    def test_reaction_creation(self, request, reaction, expected):
        """Test creation of different types of reactions, including bounds and rho_bounds"""
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
        
        # Test bounds
        assert rxn.bounds == expected["bounds"]
        # Test rho_bounds
        assert (rxn.rho_min, rxn.rho_max) == expected["rho_bounds"]

    def test_invalid_reaction_creation(self):
        """Test that creating a LabellingReaction from another LabellingReaction raises error"""
        rxn = Reaction('test')
        labelled_rxn = LabellingReaction(rxn)
        with pytest.raises(NotImplementedError):
            LabellingReaction(labelled_rxn)

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


class TestLabellingReactionAtomMapping:
    """Tests for atom mapping functionality in LabellingReaction within a model context"""

    @pytest.fixture
    def cobra_model(self, reaction_kwargs, metabolite_kwargs):
        """Create a model with all reactions from reaction_kwargs"""
        return model_builder_from_dict(reaction_kwargs, metabolite_kwargs)

    @pytest.fixture
    def model(self, cobra_model):
        return LabellingModel(LinAlg('numpy'), cobra_model)

    @pytest.mark.parametrize("r_id, pseudo, expected_atom_map", [ 
        ("r1", False, 
            {
                'A': (-1, [('a', 'b')]),
                'P': (1, [('a', 'b')])
            }
        ),
        ("r2", False, 
            {
                'A': (-1, [('a', 'b')]),
                'B': (-1, [('c', 'd')]),
                'Q': (1, [('a', 'c', 'd', 'b')])
            }
        ),
        ("r3", False, 
            {
                'A': (-2, [('a', 'b'), ('c', 'd')]),
                'Q': (1, [('a', 'c', 'd', 'b')])
            }
        ),
        ("r4", False, 
            {
                'Q': (-1, [('a', 'c', 'd', 'b')]),
                'R': (1, [('c', 'd')]),
                'S': (1, [('b', 'a')])
            }
        ),
        ("r5", False, 
            {
                'Q': (-1, [('a', 'c', 'd', 'b')]),
                'R': (2, [('c', 'd'), ('b', 'a')])
            }
        ),
        ("r6", False, 
            {
                'A': (-1, [('a', 'b')]),
                'B': (-1, [('c', 'd')]),
                'T': (1, [('a', 'c')]),
                'U': (1, [('d', 'b')])
            }
        ),
        ("pr1", True, 
            {
                'A': (-1, [('a', 'b')]),
                'B': (-1, [('c', 'd')]),
                'P': (-1, [('f', 'g')]),
                'L': (1, [('a', 'c', 'f', 'd')])
            }
        ),
    ])
    def test_reaction_atom_mapping(self, r_id, pseudo, expected_atom_map, model, reaction_kwargs, metabolite_kwargs):
        """Test atom mapping for each reaction, using the model fixture. Atoms are specified as lists of tuples."""
        # Re-add atom mapping for just the single reaction under test
        single_r_kwargs = {r_id: reaction_kwargs[r_id]}
        model.add_labelling_kwargs(single_r_kwargs, metabolite_kwargs)
        if pseudo:
            r = model.pseudo_reactions.get_by_id(r_id)
        else:
            r = model.reactions.get_by_id(r_id)
        for met_id, (stoich, atom_list) in expected_atom_map.items():
            if pseudo and (stoich > 0):
                met = model.pseudo_metabolites.get_by_id(met_id)
            else:
                met = model.metabolites.get_by_id(met_id)
            # Forward reaction
            actual_stoich, actual_arr = r.atom_map[met]
            assert stoich == actual_stoich
            expected_arr = np.array(atom_list)
            assert np.array_equal(expected_arr, actual_arr)
            # Reverse reaction (if not pseudo)
            if not pseudo:
                rev = r._rev_reaction
                actual_rev_stoich, actual_rev_arr = rev.atom_map[met]
                assert -stoich == actual_rev_stoich
                assert np.array_equal(expected_arr, actual_rev_arr)

    def test_set_atom_map_errors_all_cases(self, model, cobra_model, reaction_kwargs, metabolite_kwargs):
        """Test all error-raising conditions in LabellingReaction.set_atom_map."""

        reaction = LabellingReaction(model.reactions.get_by_id('r1'))
        other_model = LabellingModel(LinAlg('numpy'), cobra_model)
        other_model.add_labelling_kwargs(
            reaction_kwargs={'r1': reaction_kwargs['r1']}, metabolite_kwargs=metabolite_kwargs
        )
        other_reaction = other_model.reactions.get_by_id('r1')

        with pytest.raises(ValueError, match="atom_map contains non-LabelledMetabolite object"):
            reaction.set_atom_map({Metabolite('x'): (None, None)})

        with pytest.raises(ValueError, match="metabolite not in model: x"):
            reaction.set_atom_map({LabelledMetabolite(Metabolite('x')): (None, None)})

        with pytest.raises(ValueError, match="first use model.repair.*references are messed up"):
            reaction.set_atom_map(other_reaction.atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (0, np.array([['a', 'b']])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'b']])),
        }
        with pytest.raises(ValueError, match="0 stoichiometry for: A"):
            reaction.set_atom_map(atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (2, np.array([['a', 'b']])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'b']])),
        }
        with pytest.raises(ValueError, match="for A stoichiometry and atom mapping are inconsistent"):
            reaction.set_atom_map(atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (-1, np.array([[],[]])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'b']])),
        }
        with pytest.raises(ValueError, match="for A stoichiometry and atom mapping are inconsistent"):
            reaction.set_atom_map(atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (-1, np.array([['a']])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'b']])),
        }
        with pytest.raises(ValueError, match="references are messed up for: A"):
            reaction.set_atom_map(atom_map)

        reaction._metabolites = {met: stoich_atoms[0] for met, stoich_atoms in atom_map.items()}
        with pytest.raises(ValueError, match="for A different number of carbons in formula: C2H4O2, than atoms in atom mapping: C1"):
            reaction.set_atom_map(atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (-1, np.array([['a', 'b']])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'c']])),
        }
        with pytest.raises(ValueError, match="product atoms do not occur in substrate r1"):
            reaction.set_atom_map(atom_map)

        atom_map = {
            model.metabolites.get_by_id('A'): (-1, np.array([['a', 'a']])),
            model.metabolites.get_by_id('P'): (1, np.array([['a', 'b']])),
        }
        with pytest.raises(ValueError, match="non-unique atom mapping r1"):
            reaction.set_atom_map(atom_map)





        # 3. Stoichiometry and atom mapping inconsistent (stoich != model stoich)
        # with pytest.raises(ValueError, match="stoichiometry and atom mapping are inconsistent"):
        #     labelled_rxn.set_atom_map({labelled_met: (-2, [('a', 'b')]), labelled_met_B: (1, [('a', 'b')])})
        #
        # # 4. Metabolite object identity mismatch (same id, different object)
        # # (already covered by #2, but test for the specific error)
        # # To trigger the 'metabolite in atom_map is not equal to metabolite in self.metabolites'
        # # we need to pass the correct id but a different object
        # met_same_id = LabelledMetabolite(Metabolite('A', formula='C2H4O2'), formula='C2H4O2')
        # # Patch model reference to avoid the previous error
        # met_same_id._model = model
        # with pytest.raises(ValueError, match="metabolite in atom_map is not equal to metabolite in self.metabolites"):
        #     labelled_rxn.set_atom_map({met_same_id: (-1, [('a', 'b')]), labelled_met_B: (1, [('a', 'b')])})
        # # 5. Atom mapping with different number of carbons than formula
        # with pytest.raises(ValueError, match="different number of carbons in.*than atoms in atom mapping"):
        #     labelled_rxn.set_atom_map({labelled_met: (-1, [('a', 'b', 'c')]), labelled_met_B: (1, [('a', 'b')])})
        # # 6. Non-unique atom mappings (duplicate atom labels in substrates)
        # # e.g. both reactant atoms are 'a'
        # with pytest.raises(ValueError, match="non-unique atom mapping"):
        #     labelled_rxn.set_atom_map({labelled_met: (-1, [('a', 'a')]), labelled_met_B: (1, [('a', 'a')])})
        # # 7. Product atoms not present in substrates
        # with pytest.raises(ValueError, match="product atoms do not occur in substrate"):
        #     labelled_rxn.set_atom_map({labelled_met: (-1, [('a', 'b')]), labelled_met_B: (1, [('a', 'c')])})
        # # 8. Unbalanced forward/reverse reactions (different number of atoms in products vs substrates)
        # with pytest.raises(ValueError, match="cannot have a reverse reaction for an unbalanced forward reaction"):
        #     labelled_rxn.set_atom_map({labelled_met: (-1, [('a', 'b')]), labelled_met_B: (1, [('a', 'b', 'c')])})

