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

@pytest.fixture
def basic_metabolites():
    # Non-symmetric metabolites
    a = EMU_Metabolite(metabolite=Metabolite('A', formula='C3H6O3'))
    b = EMU_Metabolite(metabolite=Metabolite('B', formula='C2H4O2'))
    c = EMU_Metabolite(metabolite=Metabolite('C', formula='C2H4O2'))
    x = EMU_Metabolite(metabolite=Metabolite('X', formula='C3H6O3'))
    y = EMU_Metabolite(metabolite=Metabolite('Y', formula='C2H4O2'))
    z = EMU_Metabolite(metabolite=Metabolite('Z', formula='C3H6O3'))
    return {'a': a, 'b': b, 'c': c, 'x': x, 'y': y, 'z': z}

@pytest.fixture
def symmetric_metabolites():
    # Symmetric metabolites
    a_sym = EMU_Metabolite(metabolite=Metabolite('A_sym', formula='C3H6O3'), symmetric=True)
    b_sym = EMU_Metabolite(metabolite=Metabolite('B_sym', formula='C2H4O2'), symmetric=True)
    c_sym = EMU_Metabolite(metabolite=Metabolite('C_sym', formula='C2H4O2'), symmetric=True)
    x_sym = EMU_Metabolite(metabolite=Metabolite('X_sym', formula='C3H6O3'), symmetric=True)
    y_sym = EMU_Metabolite(metabolite=Metabolite('Y_sym', formula='C2H4O2'), symmetric=True)
    z_sym = EMU_Metabolite(metabolite=Metabolite('Z_sym', formula='C3H6O3'), symmetric=True)
    return {'a': a_sym, 'b': b_sym, 'c': c_sym, 'x': x_sym, 'y': y_sym, 'z': z_sym}

@pytest.fixture
def single_substrate_reaction(basic_metabolites):
    reaction = Reaction('A_to_X')
    reaction.add_metabolites({basic_metabolites['a']: -1, basic_metabolites['x']: 1})
    emu_reaction = EMU_Reaction(reaction=reaction)
    emu_reaction.set_atom_map({
        basic_metabolites['a']: (-1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')]),
        basic_metabolites['x']: (1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')])
    })
    return emu_reaction

@pytest.fixture
def two_substrate_one_product_reaction(basic_metabolites):
    reaction = Reaction('A_B_to_Y')
    reaction.add_metabolites({
        basic_metabolites['a']: -1,
        basic_metabolites['b']: -1,
        basic_metabolites['y']: 1
    })
    emu_reaction = EMU_Reaction(reaction=reaction)
    emu_reaction.set_atom_map({
        basic_metabolites['a']: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        basic_metabolites['b']: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        basic_metabolites['y']: (1, [('C1', 'C1'), ('C2', 'C2')])
    })
    return emu_reaction

@pytest.fixture
def two_substrate_two_product_reaction(basic_metabolites):
    reaction = Reaction('A_B_to_X_Y')
    reaction.add_metabolites({
        basic_metabolites['a']: -1,
        basic_metabolites['b']: -1,
        basic_metabolites['x']: 1,
        basic_metabolites['y']: 1
    })
    emu_reaction = EMU_Reaction(reaction=reaction)
    emu_reaction.set_atom_map({
        basic_metabolites['a']: (-1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')]),
        basic_metabolites['b']: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        basic_metabolites['x']: (1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')]),
        basic_metabolites['y']: (1, [('C1', 'C1'), ('C2', 'C2')])
    })
    return emu_reaction

@pytest.fixture
def pseudo_reaction(basic_metabolites):
    reaction = Reaction('A_B_C_to_Z')
    reaction.add_metabolites({
        basic_metabolites['a']: -1,
        basic_metabolites['b']: -1,
        basic_metabolites['c']: -1,
        basic_metabolites['z']: 1
    })
    emu_reaction = EMU_Reaction(reaction=reaction)
    emu_reaction.set_atom_map({
        basic_metabolites['a']: (-1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')]),
        basic_metabolites['b']: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        basic_metabolites['c']: (-1, [('C1', 'C1'), ('C2', 'C2')]),
        basic_metabolites['z']: (1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')])
    })
    return emu_reaction

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
        # Test default values
        assert labelled_reaction.rho_min == 0.0
        assert labelled_reaction.rho_max == 0.0

        # Test valid value assignments
        labelled_reaction.rho_max = 0.5
        assert labelled_reaction.rho_max == 0.5
        labelled_reaction.rho_min = 0.1
        assert labelled_reaction.rho_min == 0.1

        # Test values > 1.0
        with pytest.raises(ValueError, match='rho values cannot be greater than 1.0'):
            labelled_reaction.rho_max = 1.5
        with pytest.raises(ValueError, match='rho values cannot be greater than 1.0'):
            labelled_reaction.rho_min = 1.5

        # Test negative values
        with pytest.raises(ValueError, match='rho values cannot be negative'):
            labelled_reaction.rho_min = -0.1
        with pytest.raises(ValueError, match='rho values cannot be negative'):
            labelled_reaction.rho_max = -0.1

        # Test rho_min > rho_max
        labelled_reaction.rho_max = 0.3
        with pytest.raises(ValueError, match='rho_min cannot be greater than rho_max'):
            labelled_reaction.rho_min = 0.4

        # Test rho_max < _RHO_MIN
        labelled_reaction.rho_min = 0.0  # Reset rho_min first
        labelled_reaction.rho_max = 0.0005  # Less than _RHO_MIN (0.001)
        assert labelled_reaction.rho_min == 0.0
        assert labelled_reaction.rho_max == 0.0

        # Test reversible reaction (set via bounds)
        labelled_reaction.bounds = (-100, 100)  # Make reaction reversible
        labelled_reaction._dgibbsr = 0.0
        assert labelled_reaction.rho_min == 0.0
        assert labelled_reaction.rho_max == labelled_reaction._RHO_MAX

        # Test zero bounds
        labelled_reaction.bounds = (0.0, 0.0)
        assert labelled_reaction.rho_min == 0.0
        assert labelled_reaction.rho_max == 0.0

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

    def test_single_substrate_reaction(self, single_substrate_reaction, basic_metabolites):
        # Test atom mapping
        assert len(single_substrate_reaction._atom_map) == 2
        assert basic_metabolites['a'] in single_substrate_reaction._atom_map
        assert basic_metabolites['x'] in single_substrate_reaction._atom_map
        
        # Test reactant-product mapping
        product_emu = basic_metabolites['x'].get_emu(np.array([0, 1, 2]))
        mapping = single_substrate_reaction.map_reactants_products(
            product_emu=product_emu,
            substrate_metabolites=[basic_metabolites['a']]
        )
        assert len(mapping) > 0
        assert all(isinstance(m, tuple) for m in mapping)
        assert all(m[1] == product_emu for m in mapping)  # Check product EMU
        assert all(m[2].metabolite == basic_metabolites['a'] for m in mapping)  # Check reactant metabolite

        # Test reverse reaction
        rev_reaction = single_substrate_reaction._rev_reaction
        assert rev_reaction is not None
        assert rev_reaction.pseudo is True
        assert len(rev_reaction._atom_map) == 2
        assert basic_metabolites['a'] in rev_reaction._atom_map
        assert basic_metabolites['x'] in rev_reaction._atom_map
        
        # Check reverse reaction atom mapping is inverted
        for met, (stoich, atoms) in rev_reaction._atom_map.items():
            orig_stoich, orig_atoms = single_substrate_reaction._atom_map[met]
            assert stoich == -orig_stoich
            assert np.array_equal(atoms, orig_atoms)

    def test_two_substrate_one_product_reaction(self, two_substrate_one_product_reaction, basic_metabolites):
        # Test atom mapping
        assert len(two_substrate_one_product_reaction._atom_map) == 3
        assert basic_metabolites['a'] in two_substrate_one_product_reaction._atom_map
        assert basic_metabolites['b'] in two_substrate_one_product_reaction._atom_map
        assert basic_metabolites['y'] in two_substrate_one_product_reaction._atom_map
        
        # Test reactant-product mapping
        product_emu = basic_metabolites['y'].get_emu(np.array([0, 1]))
        mapping = two_substrate_one_product_reaction.map_reactants_products(
            product_emu=product_emu,
            substrate_metabolites=[basic_metabolites['a'], basic_metabolites['b']]
        )
        assert len(mapping) > 0
        assert all(isinstance(m, tuple) for m in mapping)
        assert all(m[1] == product_emu for m in mapping)  # Check product EMU
        assert all(m[2].metabolite in [basic_metabolites['a'], basic_metabolites['b']] for m in mapping)  # Check reactant metabolites

        # Test reverse reaction
        rev_reaction = two_substrate_one_product_reaction._rev_reaction
        assert rev_reaction is not None
        assert rev_reaction.pseudo is True
        assert len(rev_reaction._atom_map) == 3
        assert basic_metabolites['a'] in rev_reaction._atom_map
        assert basic_metabolites['b'] in rev_reaction._atom_map
        assert basic_metabolites['y'] in rev_reaction._atom_map
        
        # Check reverse reaction atom mapping is inverted
        for met, (stoich, atoms) in rev_reaction._atom_map.items():
            orig_stoich, orig_atoms = two_substrate_one_product_reaction._atom_map[met]
            assert stoich == -orig_stoich
            assert np.array_equal(atoms, orig_atoms)

    def test_two_substrate_two_product_reaction(self, two_substrate_two_product_reaction, basic_metabolites):
        # Test atom mapping
        assert len(two_substrate_two_product_reaction._atom_map) == 4
        assert basic_metabolites['a'] in two_substrate_two_product_reaction._atom_map
        assert basic_metabolites['b'] in two_substrate_two_product_reaction._atom_map
        assert basic_metabolites['x'] in two_substrate_two_product_reaction._atom_map
        assert basic_metabolites['y'] in two_substrate_two_product_reaction._atom_map
        
        # Test reactant-product mapping for X
        product_emu_x = basic_metabolites['x'].get_emu(np.array([0, 1, 2]))
        mapping_x = two_substrate_two_product_reaction.map_reactants_products(
            product_emu=product_emu_x,
            substrate_metabolites=[basic_metabolites['a'], basic_metabolites['b']]
        )
        assert len(mapping_x) > 0
        assert all(isinstance(m, tuple) for m in mapping_x)
        assert all(m[1] == product_emu_x for m in mapping_x)
        assert all(m[2].metabolite == basic_metabolites['a'] for m in mapping_x)  # X should only come from A
        
        # Test reactant-product mapping for Y
        product_emu_y = basic_metabolites['y'].get_emu(np.array([0, 1]))
        mapping_y = two_substrate_two_product_reaction.map_reactants_products(
            product_emu=product_emu_y,
            substrate_metabolites=[basic_metabolites['a'], basic_metabolites['b']]
        )
        assert len(mapping_y) > 0
        assert all(isinstance(m, tuple) for m in mapping_y)
        assert all(m[1] == product_emu_y for m in mapping_y)
        assert all(m[2].metabolite == basic_metabolites['b'] for m in mapping_y)  # Y should only come from B

        # Test reverse reaction
        rev_reaction = two_substrate_two_product_reaction._rev_reaction
        assert rev_reaction is not None
        assert rev_reaction.pseudo is True
        assert len(rev_reaction._atom_map) == 4
        assert basic_metabolites['a'] in rev_reaction._atom_map
        assert basic_metabolites['b'] in rev_reaction._atom_map
        assert basic_metabolites['x'] in rev_reaction._atom_map
        assert basic_metabolites['y'] in rev_reaction._atom_map
        
        # Check reverse reaction atom mapping is inverted
        for met, (stoich, atoms) in rev_reaction._atom_map.items():
            orig_stoich, orig_atoms = two_substrate_two_product_reaction._atom_map[met]
            assert stoich == -orig_stoich
            assert np.array_equal(atoms, orig_atoms)

    def test_pseudo_reaction(self, pseudo_reaction, basic_metabolites):
        # Test atom mapping
        assert len(pseudo_reaction._atom_map) == 4
        assert basic_metabolites['a'] in pseudo_reaction._atom_map
        assert basic_metabolites['b'] in pseudo_reaction._atom_map
        assert basic_metabolites['c'] in pseudo_reaction._atom_map
        assert basic_metabolites['z'] in pseudo_reaction._atom_map
        
        # Test reactant-product mapping
        product_emu = basic_metabolites['z'].get_emu(np.array([0, 1, 2]))
        mapping = pseudo_reaction.map_reactants_products(
            product_emu=product_emu,
            substrate_metabolites=[basic_metabolites['a'], basic_metabolites['b'], basic_metabolites['c']]
        )
        assert len(mapping) > 0
        assert all(isinstance(m, tuple) for m in mapping)
        assert all(m[1] == product_emu for m in mapping)
        assert all(m[2].metabolite in [basic_metabolites['a'], basic_metabolites['b'], basic_metabolites['c']] for m in mapping)

        # Test that pseudo reaction has no reverse reaction
        assert pseudo_reaction._rev_reaction is None

    def test_symmetric_metabolites(self, symmetric_metabolites):
        # Create a reaction with symmetric metabolites
        reaction = Reaction('A_sym_to_X_sym')
        reaction.add_metabolites({symmetric_metabolites['a']: -1, symmetric_metabolites['x']: 1})
        emu_reaction = EMU_Reaction(reaction=reaction)
        emu_reaction.set_atom_map({
            symmetric_metabolites['a']: (-1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')]),
            symmetric_metabolites['x']: (1, [('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3')])
        })
        
        # Test reactant-product mapping with symmetric metabolites
        product_emu = symmetric_metabolites['x'].get_emu(np.array([0, 1]))
        mapping = emu_reaction.map_reactants_products(
            product_emu=product_emu,
            substrate_metabolites=[symmetric_metabolites['a']]
        )
        assert len(mapping) > 0
        assert all(isinstance(m, tuple) for m in mapping)
        assert all(m[1] == product_emu for m in mapping)
        assert all(m[2].metabolite == symmetric_metabolites['a'] for m in mapping)
        
        # Test that symmetric metabolites generate correct number of mappings
        # For symmetric metabolites, we should get additional mappings due to symmetry
        assert len(mapping) > 1  # Should have more than one mapping due to symmetry

        # Test reverse reaction for symmetric metabolites
        rev_reaction = emu_reaction._rev_reaction
        assert rev_reaction is not None
        assert rev_reaction.pseudo is True
        assert len(rev_reaction._atom_map) == 2
        assert symmetric_metabolites['a'] in rev_reaction._atom_map
        assert symmetric_metabolites['x'] in rev_reaction._atom_map
        
        # Check reverse reaction atom mapping is inverted
        for met, (stoich, atoms) in rev_reaction._atom_map.items():
            orig_stoich, orig_atoms = emu_reaction._atom_map[met]
            assert stoich == -orig_stoich
            assert np.array_equal(atoms, orig_atoms) 