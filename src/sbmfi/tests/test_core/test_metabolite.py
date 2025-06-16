import pytest
import numpy as np
import pickle
from cobra import Metabolite
from sbmfi.core.metabolite import LabelledMetabolite, IsoCumo, EMU_Metabolite, EMU, ConvolutedEMU
from sbmfi.compound.formula import Formula

@pytest.fixture
def basic_metabolite():
    return Metabolite('test_met', formula='C6H12O6', name='Test Metabolite', charge=0, compartment='c')

@pytest.fixture
def labelled_metabolite(basic_metabolite):
    return LabelledMetabolite(metabolite=basic_metabolite, symmetric=False, formula='C6H12O6')

@pytest.fixture
def emu_metabolite(basic_metabolite):
    return EMU_Metabolite(metabolite=basic_metabolite, symmetric=False, formula='C6H12O6')

class TestLabelledMetabolite:
    def test_init_with_metabolite(self, basic_metabolite):
        lm = LabelledMetabolite(metabolite=basic_metabolite, formula='C6H12O6')
        assert lm.id == 'test_met'
        assert lm.formula == 'C6H12O6'
        assert lm.symmetric is False

    def test_init_with_labelled_metabolite(self, labelled_metabolite):
        with pytest.raises(NotImplementedError):
            LabelledMetabolite(metabolite=labelled_metabolite)

    def test_init_with_invalid_type(self):
        with pytest.raises(ValueError):
            LabelledMetabolite(metabolite="invalid")

    def test_formula_property(self, labelled_metabolite):
        assert labelled_metabolite.formula == 'C6H12O6'
        labelled_metabolite.formula = 'C5H10O5'
        assert labelled_metabolite.formula == 'C5H10O5'

    def test_formula_with_charge(self, basic_metabolite):
        with pytest.raises(ValueError):
            LabelledMetabolite(metabolite=basic_metabolite, formula='C6H12O6-')

    def test_weight_property(self, labelled_metabolite):
        assert labelled_metabolite.weight == 6  # 6 carbon atoms

    def test_formula_weight_property(self, labelled_metabolite):
        # C6H12O6 molecular weight
        assert abs(labelled_metabolite.formula_weight - 180.0633881022) < 0.001

class TestIsoCumo:
    def test_init(self, labelled_metabolite):
        iso = IsoCumo(metabolite=labelled_metabolite, label='000000', name='test_iso')
        assert iso.id == 'test_met/000000'
        assert iso.name == 'test_iso'
        assert iso.weight == 0

    def test_invalid_label(self, labelled_metabolite):
        with pytest.raises(ValueError):
            IsoCumo(metabolite=labelled_metabolite, label='123')  # Invalid characters

    def test_label_length_mismatch(self, labelled_metabolite):
        with pytest.raises(ValueError):
            IsoCumo(metabolite=labelled_metabolite, label='000')  # Wrong length

    def test_label_property(self, labelled_metabolite):
        iso = IsoCumo(metabolite=labelled_metabolite, label='000000')
        assert iso.label == '000000'
        iso.label = '111111'
        assert iso.label == '111111'
        assert iso.weight == 6

    def test_int10_property(self, labelled_metabolite):
        iso = IsoCumo(metabolite=labelled_metabolite, label='000000')
        assert iso.int10 == 0
        iso.label = '111111'
        assert iso.int10 == 63  # 2^6 - 1

class TestEMU_Metabolite:
    def test_init(self, emu_metabolite):
        assert emu_metabolite.id == 'test_met'
        assert len(emu_metabolite.emus) == 6  # One for each carbon
        assert len(emu_metabolite.convolvers) == 0

    def test_get_emu(self, emu_metabolite):
        positions = np.array([0, 1, 2])
        emu = emu_metabolite.get_emu(positions)
        assert emu.weight == 3
        assert np.array_equal(emu.positions, positions)
        assert emu.id == 'test_met|[0,1,2]'

    def test_get_emu_duplicate(self, emu_metabolite):
        positions = np.array([0, 1, 2])
        emu1 = emu_metabolite.get_emu(positions)
        emu2 = emu_metabolite.get_emu(positions)
        assert emu1 is emu2  # Should return the same object

    def test_get_convolved_emu(self, emu_metabolite):
        emu1 = emu_metabolite.get_emu(np.array([0, 1]))
        emu2 = emu_metabolite.get_emu(np.array([2, 3]))
        conv_emu = emu_metabolite.get_convolved_emu([emu1, emu2])
        assert conv_emu.weight == 4
        assert conv_emu.id == 'test_met|[0,1] ∗ test_met|[2,3]'

class TestEMU:
    def test_init(self, emu_metabolite):
        positions = np.array([0, 1, 2])
        emu = EMU(metabolite=emu_metabolite, positions=positions)
        assert emu.id == 'test_met|[0,1,2]'
        assert emu.weight == 3
        assert np.array_equal(emu.positions, positions)

    def test_invalid_positions(self, emu_metabolite):
        with pytest.raises(ValueError):
            EMU(metabolite=emu_metabolite, positions=np.array([-1, 0, 1]))  # Negative position

        with pytest.raises(ValueError):
            EMU(metabolite=emu_metabolite, positions=np.array([0, 0, 1]))  # Duplicate position

        with pytest.raises(ValueError):
            EMU(metabolite=emu_metabolite, positions=np.array([0, 1, 2, 3, 4, 5, 6]))  # Too many positions

    def test_getmu(self, emu_metabolite):
        emu = EMU(metabolite=emu_metabolite, positions=np.array([0, 1, 2]))
        mu = emu.getmu()
        assert len(mu) == 1
        assert emu in mu

class TestConvolutedEMU:
    def test_init(self, emu_metabolite):
        emu1 = EMU(metabolite=emu_metabolite, positions=np.array([0, 1]))
        emu2 = EMU(metabolite=emu_metabolite, positions=np.array([2, 3]))
        conv_emu = ConvolutedEMU(emus=[emu1, emu2])
        assert conv_emu.weight == 4
        assert conv_emu.id == 'test_met|[0,1] ∗ test_met|[2,3]'

    def test_getmu(self, emu_metabolite):
        emu1 = EMU(metabolite=emu_metabolite, positions=np.array([0, 1]))
        emu2 = EMU(metabolite=emu_metabolite, positions=np.array([2, 3]))
        conv_emu = ConvolutedEMU(emus=[emu1, emu2])
        mu = conv_emu.getmu()
        assert len(mu) == 3  # conv_emu, emu1, and emu2
        assert conv_emu in mu
        assert emu1 in mu
        assert emu2 in mu

class TestPickling:
    def test_pickle_labelled_metabolite(self, labelled_metabolite):
        # Pickle and unpickle
        pickled = pickle.dumps(labelled_metabolite)
        unpickled = pickle.loads(pickled)
        
        # Check attributes are preserved
        assert unpickled.id == labelled_metabolite.id
        assert unpickled.formula == labelled_metabolite.formula
        assert unpickled.symmetric == labelled_metabolite.symmetric
        assert unpickled.weight == labelled_metabolite.weight
        assert unpickled.formula_weight == labelled_metabolite.formula_weight

    def test_pickle_isocumo(self, labelled_metabolite):
        # Create an IsoCumo
        iso = IsoCumo(metabolite=labelled_metabolite, label='000000', name='test_iso')
        
        # Pickle and unpickle
        pickled = pickle.dumps(iso)
        unpickled = pickle.loads(pickled)
        
        # Check attributes are preserved
        assert unpickled.id == iso.id
        assert unpickled.name == iso.name
        assert unpickled.weight == iso.weight
        assert unpickled.label == iso.label
        assert unpickled.int10 == iso.int10
        # Metabolite reference should be None after unpickling
        assert unpickled.metabolite is None

    def test_pickle_emu_metabolite(self, emu_metabolite):
        # Create some EMUs and convolvers
        emu1 = emu_metabolite.get_emu(np.array([0, 1]))
        emu2 = emu_metabolite.get_emu(np.array([2, 3]))
        conv_emu = emu_metabolite.get_convolved_emu([emu1, emu2])
        
        # Pickle and unpickle
        pickled = pickle.dumps(emu_metabolite)
        unpickled = pickle.loads(pickled)
        
        # Check basic attributes
        assert unpickled.id == emu_metabolite.id
        assert unpickled.formula == emu_metabolite.formula
        assert unpickled.symmetric == emu_metabolite.symmetric
        
        # Check EMUs structure is recreated
        assert len(unpickled.emus) == len(emu_metabolite.emus)
        
        # Check all EMU lists are empty except for the full metabolite EMU
        full_weight = emu_metabolite.elements['C']
        for weight in unpickled.emus:
            if weight == full_weight:
                assert len(unpickled.emus[weight]) == 1  # Should have the self-EMU
                assert unpickled.emus[weight][0].id == f"{unpickled.id}|[0,1,2,3,4,5]"  # Full carbon positions
            else:
                assert len(unpickled.emus[weight]) == 0  # Other EMUs should be empty
        
        # Check convolvers are cleared
        assert len(unpickled.convolvers) == 0

    def test_pickle_emu(self, emu_metabolite):
        # Create an EMU
        positions = np.array([0, 1, 2])
        emu = EMU(metabolite=emu_metabolite, positions=positions)
        
        # Pickle and unpickle
        pickled = pickle.dumps(emu)
        unpickled = pickle.loads(pickled)
        
        # Check attributes are preserved
        assert unpickled.id == emu.id
        assert unpickled.weight == emu.weight
        assert np.array_equal(unpickled.positions, emu.positions)
        # Metabolite reference should be None after unpickling
        assert unpickled.metabolite is None

    def test_pickle_convoluted_emu(self, emu_metabolite):
        # Create EMUs and a ConvolutedEMU
        emu1 = EMU(metabolite=emu_metabolite, positions=np.array([0, 1]))
        emu2 = EMU(metabolite=emu_metabolite, positions=np.array([2, 3]))
        conv_emu = ConvolutedEMU(emus=[emu1, emu2])
        
        # Pickle and unpickle
        pickled = pickle.dumps(conv_emu)
        unpickled = pickle.loads(pickled)
        
        # Check attributes are preserved
        assert unpickled.id == conv_emu.id
        assert unpickled.weight == conv_emu.weight
        # EMUs should be preserved
        assert len(unpickled._emus) == len(conv_emu._emus)
        for e1, e2 in zip(unpickled._emus, conv_emu._emus):
            assert e1.id == e2.id
            assert e1.weight == e2.weight 