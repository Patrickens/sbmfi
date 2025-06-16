import pytest
import numpy as np
import pandas as pd
from cobra import Model, Reaction, Metabolite
from sbmfi.core.model import LabellingModel, EMU_Model, create_full_metabolite_kwargs, model_builder_from_dict
from sbmfi.core.linalg import LinAlg
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU, IsoCumo
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction

@pytest.fixture
def basic_model():
    model = Model('test_model')
    return model

@pytest.fixture
def linalg():
    return LinAlg(backend='numpy')

@pytest.fixture
def labelled_model(basic_model, linalg):
    return LabellingModel(linalg=linalg, model=basic_model)

@pytest.fixture
def emu_model(basic_model, linalg):
    return EMU_Model(linalg=linalg, model=basic_model)

@pytest.fixture
def simple_reaction():
    reaction = Reaction('test_rxn')
    a = Metabolite('A', formula='C6H12O6')
    b = Metabolite('B', formula='C6H12O6')
    reaction.add_metabolites({a: -1, b: 1})
    return reaction

@pytest.fixture
def labelled_reaction(simple_reaction):
    lr = LabellingReaction(reaction=simple_reaction)
    a = list(lr.metabolites.keys())[0]
    b = list(lr.metabolites.keys())[1]
    lr.set_atom_map({
        a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        b: (1, [('C1', 'C1'), ('C2', 'C2')])
    })
    return lr

@pytest.fixture
def emu_reaction(simple_reaction):
    er = EMU_Reaction(reaction=simple_reaction)
    a = list(er.metabolites.keys())[0]
    b = list(er.metabolites.keys())[1]
    er.set_atom_map({
        a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        b: (1, [('C1', 'C1'), ('C2', 'C2')])
    })
    return er

class TestLabellingModel:
    def test_init_with_model(self, basic_model, linalg):
        lm = LabellingModel(linalg=linalg, model=basic_model)
        assert lm.id == 'test_model'
        assert lm._is_built is False
        assert lm._la is linalg

    def test_init_with_labelling_model(self, labelled_model, linalg):
        with pytest.raises(NotImplementedError):
            LabellingModel(linalg=linalg, model=labelled_model)

    def test_init_with_invalid_type(self, linalg):
        with pytest.raises(ValueError):
            LabellingModel(linalg=linalg, model="invalid")

    def test_properties(self, labelled_model):
        assert labelled_model.is_built is False
        assert labelled_model.biomass_id == ''
        assert labelled_model.labelling_id == ''
        assert len(labelled_model.measurements) == 0
        assert len(labelled_model.labelling_reactions) == 0

    def test_set_substrate_labelling(self, labelled_model, labelled_reaction):
        # Add reaction to model
        labelled_model.add_reactions([labelled_reaction])
        
        # Create substrate labelling
        a = list(labelled_reaction.metabolites.keys())[0]
        iso = IsoCumo(metabolite=a, label='00')
        substrate_labelling = pd.Series({iso.id: 1.0}, name='test_labelling')
        
        # Test setting substrate labelling
        labelled_model.set_substrate_labelling(substrate_labelling)
        assert labelled_model.labelling_id == 'test_labelling'
        assert len(labelled_model._substrate_labelling) == 1

    def test_set_measurements(self, labelled_model, labelled_reaction):
        # Add reaction to model
        labelled_model.add_reactions([labelled_reaction])
        
        # Test setting measurements
        measurement_list = ['A']
        labelled_model.set_measurements(measurement_list)
        assert len(labelled_model.measurements) == 1
        assert labelled_model.measurements[0].id == 'A'

    def test_add_remove_reactions(self, labelled_model, labelled_reaction):
        # Test adding reaction
        labelled_model.add_reactions([labelled_reaction])
        assert len(labelled_model.reactions) == 1
        assert labelled_model.reactions[0].id == 'test_rxn'

        # Test removing reaction
        labelled_model.remove_reactions([labelled_reaction])
        assert len(labelled_model.reactions) == 0

    def test_add_remove_metabolites(self, labelled_model, labelled_reaction):
        # Add reaction to model first
        labelled_model.add_reactions([labelled_reaction])
        
        # Test removing metabolites
        metabolites = list(labelled_reaction.metabolites.keys())
        labelled_model.remove_metabolites(metabolites)
        assert len(labelled_model.metabolites) == 0

    def test_copy(self, labelled_model, labelled_reaction):
        # Add reaction to model
        labelled_model.add_reactions([labelled_reaction])
        
        # Test copying model
        copied_model = labelled_model.copy()
        assert copied_model.id == labelled_model.id
        assert len(copied_model.reactions) == len(labelled_model.reactions)
        assert copied_model._is_built is False

class TestEMU_Model:
    def test_init(self, basic_model, linalg):
        em = EMU_Model(linalg=linalg, model=basic_model)
        assert em.id == 'test_model'
        assert em._is_built is False
        assert em._la is linalg

    def test_set_substrate_labelling(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Create substrate labelling
        a = list(emu_reaction.metabolites.keys())[0]
        emu = a.get_emu(np.array([0, 1]))
        substrate_labelling = pd.Series({emu.id: 1.0}, name='test_labelling')
        
        # Test setting substrate labelling
        emu_model.set_substrate_labelling(substrate_labelling)
        assert emu_model.labelling_id == 'test_labelling'
        assert len(emu_model._substrate_labelling) == 1

    def test_set_measurements(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Test setting measurements
        measurement_list = ['A']
        emu_model.set_measurements(measurement_list)
        assert len(emu_model.measurements) == 1
        assert emu_model.measurements[0].id == 'A'

    def test_build_model(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Set measurements
        emu_model.set_measurements(['A'])
        
        # Test building model
        emu_model.build_model()
        assert emu_model._is_built is True

    def test_cascade(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Set measurements
        emu_model.set_measurements(['A'])
        
        # Build model
        emu_model.build_model()
        
        # Test cascade
        cascade = emu_model.cascade()
        assert isinstance(cascade, pd.DataFrame)
        assert len(cascade) > 0

    def test_pretty_cascade(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Set measurements
        emu_model.set_measurements(['A'])
        
        # Build model
        emu_model.build_model()
        
        # Test pretty cascade
        pretty_str = emu_model.pretty_cascade(weight=2)
        assert isinstance(pretty_str, str)
        assert len(pretty_str) > 0

    def test_dsdv(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Set measurements
        emu_model.set_measurements(['A'])
        
        # Build model
        emu_model.build_model()
        
        # Test dsdv
        dsdv = emu_model.dsdv(emu_reaction)
        assert isinstance(dsdv, np.ndarray)
        assert dsdv.shape[0] > 0

    def test_compute_jacobian(self, emu_model, emu_reaction):
        # Add reaction to model
        emu_model.add_reactions([emu_reaction])
        
        # Set measurements
        emu_model.set_measurements(['A'])
        
        # Build model
        emu_model.build_model()
        
        # Test computing jacobian
        emu_model.compute_jacobian()
        assert emu_model._jacobian is not None 

class TestModelBuilder:
    """Test suite for model building functions"""

    @pytest.fixture
    def basic_reaction_kwargs(self):
        return {
            'r1': {
                'atom_map_str': 'A/a + B/b --> C/ab',
                'upper_bound': 100.0,
                'lower_bound': 0.0
            }
        }

    @pytest.fixture
    def basic_metabolite_kwargs(self):
        return {
            'A': {'formula': 'C1', 'compartment': 'c', 'charge': -1},
            'B': {'formula': 'C1', 'compartment': 'c', 'charge': -1},
            'C': {'formula': 'C2', 'compartment': 'c', 'charge': -2}
        }

    def test_basic_model_creation(self, basic_reaction_kwargs, basic_metabolite_kwargs):
        """Test basic model creation with simple reactions and metabolites"""
        model = model_builder_from_dict(basic_reaction_kwargs, basic_metabolite_kwargs)
        
        assert model.id == 'model'
        assert len(model.reactions) == 1
        assert len(model.metabolites) == 3
        assert 'r1' in model.reactions
        assert all(m in model.metabolites for m in ['A', 'B', 'C'])
        
        # Verify metabolite attributes
        for met_id, kwargs in basic_metabolite_kwargs.items():
            metabolite = model.metabolites.get_by_id(met_id)
            assert metabolite.formula == kwargs['formula']
            assert metabolite.compartment == kwargs['compartment']
            assert metabolite.charge == kwargs['charge']

    def test_biomass_reaction(self, basic_metabolite_kwargs):
        """Test model creation with biomass reaction"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a --> B/a'},
            'biomass': {
                'reaction_str': '0.1 A + 0.6 B --> ',
                'upper_bound': 1000.0,
                'lower_bound': 0.0
            }
        }
        
        model = model_builder_from_dict(reaction_kwargs, basic_metabolite_kwargs)
        
        assert 'biomass' in model.reactions
        biomass = model.reactions.get_by_id('biomass')
        assert biomass.metabolites[model.metabolites.get_by_id('A')] == -0.1
        assert biomass.metabolites[model.metabolites.get_by_id('B')] == -0.6

    def test_reaction_bounds(self, basic_metabolite_kwargs):
        """Test reaction bounds are properly set"""
        reaction_kwargs = {
            'r1': {
                'atom_map_str': 'A/a --> B/a',
                'upper_bound': 50.0,
                'lower_bound': -10.0
            }
        }
        
        model = model_builder_from_dict(reaction_kwargs, basic_metabolite_kwargs)
        reaction = model.reactions.get_by_id('r1')
        
        assert reaction.upper_bound == 50.0
        assert reaction.lower_bound == -10.0

    def test_metabolite_attributes(self, basic_reaction_kwargs):
        """Test metabolite attributes are properly set"""
        metabolite_kwargs = {
            'A': {
                'formula': 'C1H2O',
                'name': 'Metabolite A',
                'charge': -1,
                'symmetric': -1,
                'compartment': 'c'
            },
            'B': {
                'formula': 'C1H2O',
                'name': 'Metabolite B',
                'charge': -1,
                'symmetric': -1,
                'compartment': 'c'
            }
        }
        
        model = model_builder_from_dict(basic_reaction_kwargs, metabolite_kwargs)
        metabolite_a = model.metabolites.get_by_id('A')
        
        assert metabolite_a.formula == 'C1H2O'
        assert metabolite_a.name == 'Metabolite A'
        assert metabolite_a.charge == -1
        assert metabolite_a.compartment == 'c'

    def test_empty_inputs(self):
        """Test model creation with empty inputs"""
        model = model_builder_from_dict({}, {})
        assert len(model.reactions) == 0
        assert len(model.metabolites) == 0

    def test_invalid_atom_mapping(self):
        """Test handling of invalid atom mapping"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a --> B/ab'}  # Invalid: B should have 1 atom but has 2
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        with pytest.raises(ValueError, match="Formula mismatch for metabolite B: formula has 1 carbon atoms but atom mapping has 2 atoms"):
            model_builder_from_dict(reaction_kwargs, metabolite_kwargs)

    def test_create_full_metabolite_kwargs_basic(self):
        """Test basic metabolite kwargs creation"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + B/b --> C/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'
        assert result['C']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_infer_formula(self):
        """Test formula inference"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + B/b --> C/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs, infer_formula=True)
        assert result['C']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_no_infer(self):
        """Test behavior when infer_formula=False"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + B/b --> C/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs, infer_formula=False)
        assert result['C']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_add_cofactors(self):
        """Test behavior when add_cofactors=True"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + B/b --> C/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs, add_cofactors=True)
        assert 'C' in result
        assert isinstance(result['C'], dict)

    def test_create_full_metabolite_kwargs_empty(self):
        """Test behavior with empty inputs"""
        result = create_full_metabolite_kwargs({}, {})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_create_full_metabolite_kwargs_biomass(self):
        """Test handling of biomass reactions"""
        reaction_kwargs = {
            'biomass': {'reaction_str': '0.1 A + 0.6 B --> '}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'

    def test_create_full_metabolite_kwargs_empty_metabolite(self):
        """Test handling of empty metabolite (∅) in reaction string"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + ∅ --> B/a'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert '∅' not in result  # Should not add empty metabolite

    def test_create_full_metabolite_kwargs_reaction_str(self):
        """Test handling of reaction_str instead of atom_map_str"""
        reaction_kwargs = {
            'r1': {'reaction_str': 'A + B --> C'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'
        assert result['C']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_complex_reaction(self):
        """Test handling of complex reaction with multiple metabolites"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + B/b + C/c --> D/abc + E/d'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C1'},
            'D': {'formula': 'C3'},
            'E': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert all(m in result for m in ['A', 'B', 'C', 'D', 'E'])
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'
        assert result['C']['formula'] == 'C1'
        assert result['D']['formula'] == 'C3'
        assert result['E']['formula'] == 'C1'

    def test_create_full_metabolite_kwargs_multiple_reactions(self):
        """Test handling of multiple reactions in reaction_kwargs"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a --> B/a'},
            'r2': {'atom_map_str': 'B/b --> C/b'},
            'r3': {'atom_map_str': 'C/c --> D/c'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C1'},
            'D': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert all(m in result for m in ['A', 'B', 'C', 'D'])
        assert all(result[met]['formula'] == 'C1' for met in ['A', 'B', 'C', 'D'])

    def test_create_full_metabolite_kwargs_reversible_reaction(self):
        """Test handling of reversible reactions"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a <--> B/a'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'

    def test_create_full_metabolite_kwargs_metabolite_without_atom_mapping(self):
        """Test handling of metabolites without atom mapping in reaction string"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A + B/b --> C/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'
        assert result['C']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_multiple_atom_mappings(self):
        """Test handling of multiple atom mappings for same metabolite"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + A/b --> B/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C2'

    def test_create_full_metabolite_kwargs_cofactor_with_atom_mapping(self):
        """Test handling of cofactor with atom mapping"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a + cof/b --> B/ab'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert 'A' in result
        assert 'B' in result
        assert 'cof' in result
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C2'
        assert isinstance(result['cof'], dict)

    def test_create_full_metabolite_kwargs_invalid_atom_mapping(self):
        """Test handling of invalid atom mapping format"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a --> B/ab'}  # Invalid: B should have 1 atom but has 2
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'}
        }
        
        with pytest.raises(ValueError, match="Formula mismatch for metabolite B: formula has 1 carbon atoms but atom mapping has 2 atoms"):
            create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)

    def test_create_full_metabolite_kwargs_mixed_reaction_formats(self):
        """Test handling of mixed reaction formats (atom_map_str and reaction_str)"""
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/a --> B/a'},
            'r2': {'reaction_str': 'B + C --> D'}
        }
        metabolite_kwargs = {
            'A': {'formula': 'C1'},
            'B': {'formula': 'C1'},
            'C': {'formula': 'C1'},
            'D': {'formula': 'C2'}
        }
        
        result = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)
        assert all(m in result for m in ['A', 'B', 'C', 'D'])
        assert result['A']['formula'] == 'C1'
        assert result['B']['formula'] == 'C1'
        assert result['C']['formula'] == 'C1'
        assert result['D']['formula'] == 'C2' 