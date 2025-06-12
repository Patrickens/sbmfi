import pytest
import numpy as np
from cobra import Reaction, Metabolite
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU

@pytest.fixture
def basic_reaction():
    reaction = Reaction('test_rxn')
    reaction.add_metabolites({
        Metabolite('A', formula='C6H12O6'): -1,
        Metabolite('B', formula='C6H12O6'): 1
    })
    return reaction

@pytest.fixture
def labelled_reaction(basic_reaction):
    return LabellingReaction(reaction=basic_reaction)

@pytest.fixture
def emu_reaction(basic_reaction):
    return EMU_Reaction(reaction=basic_reaction)

@pytest.fixture
def atom_mapped_reaction():
    reaction = Reaction('test_rxn')
    a = Metabolite('A', formula='C6H12O6')
    b = Metabolite('B', formula='C6H12O6')
    reaction.add_metabolites({a: -1, b: 1})
    lr = LabellingReaction(reaction=reaction)
    lr.set_atom_map({
        a: (-1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3'), ('C4', 'C4'), ('C5', 'C5'), ('C6', 'C6')]),
        b: (1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3'), ('C4', 'C4'), ('C5', 'C5'), ('C6', 'C6')])
    })
    return lr

class TestLabellingReaction:
    def test_init_with_reaction(self, basic_reaction):
        lr = LabellingReaction(reaction=basic_reaction)
        assert lr.id == 'test_rxn'
        assert len(lr.metabolites) == 2
        assert lr._pseudo is False
        assert lr._rho_min == 0.0
        assert lr._rho_max == 0.0

    def test_init_with_labelling_reaction(self, labelled_reaction):
        with pytest.raises(NotImplementedError):
            LabellingReaction(reaction=labelled_reaction)

    def test_init_with_invalid_type(self):
        with pytest.raises(ValueError):
            LabellingReaction(reaction="invalid")

    def test_pseudo_property(self, labelled_reaction):
        assert labelled_reaction.pseudo is False
        labelled_reaction.pseudo = True
        assert labelled_reaction.pseudo is True

    def test_rho_bounds(self, labelled_reaction):
        # Test rho_max
        labelled_reaction.rho_max = 0.5
        assert labelled_reaction.rho_max == 0.5
        with pytest.raises(ValueError):
            labelled_reaction.rho_max = 1.5  # Should be <= 0.999

        # Test rho_min
        labelled_reaction.rho_min = 0.1
        assert labelled_reaction.rho_min == 0.1
        with pytest.raises(ValueError):
            labelled_reaction.rho_min = -0.1  # Should be >= 0

    def test_dgibbsr_property(self, labelled_reaction):
        labelled_reaction.dgibbsr = -10.0  # -10 kJ/mol
        assert labelled_reaction.dgibbsr == -10.0

    def test_set_atom_map(self, labelled_reaction):
        a = list(labelled_reaction.metabolites.keys())[0]
        b = list(labelled_reaction.metabolites.keys())[1]
        atom_map = {
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        }
        labelled_reaction.set_atom_map(atom_map)
        assert len(labelled_reaction._atom_map) == 2

    def test_build_reaction_string(self, atom_mapped_reaction):
        rxn_str = atom_mapped_reaction.build_reaction_string()
        assert 'A/C1C1' in rxn_str
        assert 'B/C1C1' in rxn_str
        assert '-->' in rxn_str

    def test_multiply_reaction(self, atom_mapped_reaction):
        original_str = atom_mapped_reaction.build_reaction_string()
        atom_mapped_reaction *= -1
        new_str = atom_mapped_reaction.build_reaction_string()
        assert original_str != new_str
        assert 'A/C1C1' in new_str
        assert 'B/C1C1' in new_str

    def test_add_reaction(self, labelled_reaction):
        with pytest.raises(ValueError):
            labelled_reaction + labelled_reaction

class TestEMU_Reaction:
    def test_init(self, basic_reaction):
        er = EMU_Reaction(reaction=basic_reaction)
        assert er.id == 'test_rxn'
        assert len(er.metabolites) == 2
        assert er._pseudo is False

    def test_find_reactant_emus(self, emu_reaction):
        # Create test metabolites and EMUs
        a = EMU_Metabolite(metabolite=Metabolite('A', formula='C6H12O6'))
        b = EMU_Metabolite(metabolite=Metabolite('B', formula='C6H12O6'))
        emu_reaction.add_metabolites({a: -1, b: 1})
        
        # Create atom mapping
        emu_reaction.set_atom_map({
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        })

        # Create product EMU
        product_emu = b.get_emu(np.array([0, 1]))
        
        # Test finding reactant EMUs
        reactant_emus = emu_reaction._find_reactant_emus(
            product_emu=product_emu,
            substrate_metabolites=[a]
        )
        assert len(reactant_emus) > 0
        assert all(isinstance(emu, EMU) for emu in reactant_emus)

    def test_product_elements(self, emu_reaction):
        # Create test metabolites and EMUs
        a = EMU_Metabolite(metabolite=Metabolite('A', formula='C6H12O6'))
        b = EMU_Metabolite(metabolite=Metabolite('B', formula='C6H12O6'))
        emu_reaction.add_metabolites({a: -1, b: 1})
        
        # Create atom mapping
        emu_reaction.set_atom_map({
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        })

        # Create product EMU
        product_emu = b.get_emu(np.array([0, 1]))
        
        # Test getting product elements
        elements = emu_reaction._product_elements(product_emu)
        assert len(elements) > 0
        assert all(isinstance(elem, tuple) for elem in elements)

    def test_map_reactants_products(self, emu_reaction):
        # Create test metabolites and EMUs
        a = EMU_Metabolite(metabolite=Metabolite('A', formula='C6H12O6'))
        b = EMU_Metabolite(metabolite=Metabolite('B', formula='C6H12O6'))
        emu_reaction.add_metabolites({a: -1, b: 1})
        
        # Create atom mapping
        emu_reaction.set_atom_map({
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        })

        # Create product EMU
        product_emu = b.get_emu(np.array([0, 1]))
        
        # Test mapping reactants to products
        mapping = emu_reaction.map_reactants_products(
            product_emu=product_emu,
            substrate_metabolites=[a]
        )
        assert len(mapping) > 0
        assert all(isinstance(m, tuple) for m in mapping)

    def test_build_tensors(self, emu_reaction):
        # Create test metabolites and EMUs
        a = EMU_Metabolite(metabolite=Metabolite('A', formula='C6H12O6'))
        b = EMU_Metabolite(metabolite=Metabolite('B', formula='C6H12O6'))
        emu_reaction.add_metabolites({a: -1, b: 1})
        
        # Create atom mapping
        emu_reaction.set_atom_map({
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        })

        # Test building tensors
        tensors = emu_reaction.build_tensors()
        assert len(tensors) > 0
        assert all(isinstance(t, dict) for t in tensors)

    def test_pretty_tensors(self, emu_reaction):
        # Create test metabolites and EMUs
        a = EMU_Metabolite(metabolite=Metabolite('A', formula='C6H12O6'))
        b = EMU_Metabolite(metabolite=Metabolite('B', formula='C6H12O6'))
        emu_reaction.add_metabolites({a: -1, b: 1})
        
        # Create atom mapping
        emu_reaction.set_atom_map({
            a: (-1, [('C1', 'C1'), ('C2', 'C2')]),
            b: (1, [('C1', 'C1'), ('C2', 'C2')])
        })

        # Test pretty printing tensors
        pretty_str = emu_reaction.pretty_tensors(weight=2)
        assert isinstance(pretty_str, str)
        assert len(pretty_str) > 0 