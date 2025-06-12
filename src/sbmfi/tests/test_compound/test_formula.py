import pytest
import numpy as np
from sbmfi.compound import Formula, isotopologues

def test_formula_initialization():
    # Test empty formula
    f = Formula()
    assert len(f) == 0

    # Test formula from string
    f = Formula('C6H12O6')
    assert f['C'] == 6
    assert f['H'] == 12
    assert f['O'] == 6

    # Test formula with charge
    f = Formula('C6H12O6-')
    assert f['-'] == 1

    # Test formula with isotopes
    f = Formula('[13]C6H12O6')
    assert f['[13]C'] == 6
    assert f['H'] == 12
    assert f['O'] == 6

    # Test invalid formula
    with pytest.raises(ValueError):
        Formula('Invalid')

def test_formula_arithmetic():
    f1 = Formula('C6H12O6')
    f2 = Formula('H2O')

    # Test addition
    f3 = f1 + f2
    assert f3['C'] == 6
    assert f3['H'] == 14
    assert f3['O'] == 7

    # Test subtraction
    f3 = f1 - f2
    assert f3['C'] == 6
    assert f3['H'] == 10
    assert f3['O'] == 5

    # Test multiplication
    f3 = f1 * 2
    assert f3['C'] == 12
    assert f3['H'] == 24
    assert f3['O'] == 12

def test_formula_mass():
    # Test monoisotopic mass
    f = Formula('C6H12O6')
    assert abs(f.mass() - 180.063388) < 1e-6

    # Test average mass
    assert abs(f.mass(average=True) - 180.1559) < 1e-4

    # Test with different abundances
    custom_abundances = {
        'C': {0: (12.0, 0.989), 13: (13.003355, 0.011)},
        'H': {0: (1.007825, 0.999885), 2: (2.014102, 0.000115)},
        'O': {0: (15.994915, 0.99762), 17: (16.999131, 0.00038), 18: (17.999160, 0.002)},
        '-': {0: (0.000549, 1.0)}
    }
    assert abs(f.mass(abundances=custom_abundances) - 180.063388) < 1e-6

def test_formula_mz():
    # Test basic m/z calculation
    f = Formula('C6H12O6')
    assert abs(f.mz(electrons=1) - 179.0553) < 1e-4

    # Test with negative charge
    f = Formula('C6H12O6-')
    assert abs(f.mz() - 179.0553) < 1e-4

    # Test with multiple electrons
    f = Formula('C6H12O6')
    assert abs(f.mz(electrons=2) - 89.5276) < 1e-4

    # Test with isotopes
    f = Formula('[13]C6H12O6')
    assert abs(f.mz(electrons=1) - 185.0683) < 1e-4

    # Test error for uncharged molecule
    f = Formula('C6H12O6')
    with pytest.raises(ValueError, match='Result is not a charged molecule!'):
        f.mz()

def test_formula_isotopes():
    f = Formula('C6H12O6')
    
    # Test adding C13
    f.add_C13(2)
    assert f['[13]C'] == 2
    assert f['C'] == 4

    # Test full isotope
    f = Formula('C6H12O6')
    f_full = f.full_isotope()
    assert all(k.startswith('[') for k in f_full.keys() if k != '-')

    # Test no isotope
    f = Formula('[13]C6H12O6')
    f_no = f.no_isotope()
    assert 'C' in f_no
    assert '[13]C' not in f_no

def test_formula_abundance():
    f = Formula('C6H12O6')
    assert abs(f.abundance() - 1.0) < 1e-10

    f = Formula('[13]C6H12O6')
    assert f.abundance() < 1.0

def test_formula_shift():
    f = Formula('C6H12O6')
    assert f.shift() == 0.0

    f = Formula('[13]C6H12O6')
    assert f.shift() > 0.0

def test_formula_to_chnops():
    f = Formula('C6H12O6')
    assert f.to_chnops() == 'C6H12O6'

    f = Formula('[13]C6H12O6')
    assert f.to_chnops() == '[13]C6H12O6'

    f = Formula('C6H12O6-')
    assert f.to_chnops() == 'C6H12O6-'

def test_isotopologues():
    f = Formula('C6H12O6')
    
    # Test basic isotopologues
    isos = isotopologues(f)
    assert len(isos) > 1
    
    # Test with abundance reporting
    isos_with_abundance = isotopologues(f, report_abundance=True)
    assert all(isinstance(x, tuple) for x in isos_with_abundance)
    assert all(len(x) == 2 for x in isos_with_abundance)
    
    # Test with specific elements
    isos = isotopologues(f, elements_with_isotopes=['C'])
    assert all('H' not in iso for iso in isos)
    
    # Test with abundance threshold
    isos = isotopologues(f, isotope_threshold=0.1)
    assert len(isos) < len(isotopologues(f))
    
    # Test with mass difference
    isos = isotopologues(f, n_mdv=1)
    assert all(iso.shift() == 1 for iso in isos)

def test_formula_equality():
    f1 = Formula('C6H12O6')
    f2 = Formula('C6H12O6')
    f3 = Formula('C6H12O6-')
    
    assert f1 == f2
    assert f1 != f3

def test_formula_copy():
    f1 = Formula('C6H12O6')
    f2 = f1.copy()
    
    assert f1 == f2
    assert f1 is not f2
    
    f2['C'] = 5
    assert f1['C'] == 6
    assert f2['C'] == 5 