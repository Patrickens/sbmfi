import pytest
import numpy as np
import pandas as pd
from cobra import Model, Reaction, Metabolite
from sbmfi.core.model import LabellingModel, EMU_Model
from sbmfi.core.linalg import LinAlg
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU, IsoCumo
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction

@pytest.fixture
def basic_model():
    model = Model('test_model')
    return model

@pytest.fixture
def linalg():
    return LinAlg()

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