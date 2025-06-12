import pytest
import numpy as np
from sbmfi.compound import Formula, isotopologues
from sbmfi.compound.formula import _nist_mass

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

    # Test formula with multiple isotopes of same element
    f = Formula('[13]C2[12]C4H12O6')
    assert f['[13]C'] == 2
    assert f['[12]C'] == 4
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
    assert abs(f.mass() - 180.0633881022) < 1e-6

    # Test average mass  180.1561339901187
    assert abs(f.mass(average=True) - 180.1561339901187) < 1e-4

    # Test with different abundances
    custom_abundances = {
        'C': {0: (13.003355, 0.989), 13: (12.0, 0.011)},  # switched the 13C and 12C masses
        'H': {0: (1.007825, 0.999885), 2: (2.014102, 0.000115)},
        'O': {0: (15.994915, 0.99762), 17: (16.999131, 0.00038), 18: (17.999160, 0.002)},
        '-': {0: (0.000549, 1.0)}
    }
    assert abs(f.mass(abundances=custom_abundances) - 186.08352000000002) < 1e-6

def test_formula_mz():
    # Test basic m/z calculation
    f = Formula('C6H12O6')
    mass_2 = 90.03224263100944
    assert abs(f.mz(electrons=2) - mass_2) < 1e-4

    # Test with negative charge
    f = Formula('C6H12O6-2')
    assert abs(f.mz() - mass_2) < 1e-4

    # Test with isotopes
    f = Formula('[13]C6H12O6')
    assert abs(f.mz(electrons=1) - 186.0840657089094) < 1e-4

    # Test error for uncharged molecule
    f = Formula('C6H12O6')
    with pytest.raises(ValueError, match='Result is not a charged molecule!'):
        f.mz()

def test_formula_isotopes():
    f = Formula('C6H12O6')
    
    # Test adding C13
    f_new = f.add_C13(2)
    assert f_new['[13]C'] == 2
    assert f_new['[12]C'] == 4
    assert f_new['H'] == 12
    assert f_new['O'] == 6

    # Test full isotope
    f = Formula('C6H12O6')
    f_full = f.full_isotope()
    # Check that all atoms are isotopically labeled
    assert all(k.startswith('[') for k in f_full.keys() if k != '-')
    # Check that the counts are preserved
    assert sum(f_full.values()) == sum(f.values())
    # Check that the most abundant isotopes are used
    assert f_full['[12]C'] == 6  # Most abundant C isotope
    assert f_full['[1]H'] == 12  # Most abundant H isotope
    assert f_full['[16]O'] == 6  # Most abundant O isotope

    # Test with already labeled atoms
    f = Formula('[13]C6H12O6')
    f_full = f.full_isotope()
    assert f_full['[13]C'] == 6  # Should preserve existing labels
    assert f_full['[1]H'] == 12  # Should label unlabeled atoms
    assert f_full['[16]O'] == 6

    # Test no isotope
    f = Formula('[13]C6H12O6')
    f_no = f.no_isotope()
    assert 'C' in f_no
    assert '[13]C' not in f_no
    assert f_no['C'] == 6
    assert f_no['H'] == 12
    assert f_no['O'] == 6

def test_formula_abundance():
    # Test unlabeled formula
    f = Formula('C6H12O6')
    assert abs(f.abundance() - 0.9226329790503718) < 1e-6

    # Test other abundances
    no_C = _nist_mass.copy()
    no_C.pop('C')
    assert (f.abundance(no_C) - 0.9841491454794983) < 1e-6

    # Test labeled formula
    f = Formula('[13]CC5H12O6')
    assert (f.abundance() - 0.0598736856919376) < 1e-6

def test_formula_isotope_number():
    # Test unlabeled formula
    f = Formula('C6H12O6')
    assert f.isotope_number() == 0
    assert f.isotope_number(relative=False) == 0

    # Test single isotope
    f = Formula('[13]C6H12O6')
    assert f.isotope_number() == 6  # 6 * (13-12)
    assert f.isotope_number(relative=False) == 78  # 6 * 13

    # Test multiple isotopes
    f = Formula('[13]C2[12]C4H12O6')
    assert f.isotope_number() == 2  # 2 * (13-12)
    assert f.isotope_number(relative=False) == 26  # 2 * 13

    # Test with charge
    f = Formula('[13]C6H12O6-')
    assert f.isotope_number() == 6  # charge should not affect the result
    assert f.isotope_number(relative=False) == 78

def test_formula_to_chnops():
    f = Formula('C6H12O6')
    assert f.to_chnops() == 'C6H12O6'

    f = Formula('[13]C6H12O6')
    assert f.to_chnops() == '[13]C6H12O6'

    f = Formula('C6H12O6-')
    assert f.to_chnops() == 'C6H12O6-'

    f = Formula('[13]C2[12]C4H12O6')
    assert f.to_chnops() == '[12]C4[13]C2H12O6'  # Should be sorted by isotope number

def test_isotopologues():
    f = Formula('C2HO')

    # Test basic isotopologues
    check = eval(
        "[Formula({'[12]C': 2, '[1]H': 1, '[16]O': 1}), Formula({'[12]C': 2, '[1]H': 1, '[18]O': 1}), "
        "Formula({'[12]C': 1, '[13]C': 1, '[1]H': 1, '[16]O': 1}), "
        "Formula({'[12]C': 1, '[13]C': 1, '[1]H': 1, '[18]O': 1}), "
        "Formula({'[13]C': 2, '[1]H': 1, '[16]O': 1}), Formula({'[13]C': 2, '[1]H': 1, '[18]O': 1})]"
    )
    isos = list(isotopologues(f))
    assert isos == check

    # Test with specific elements
    check = eval(
        "[Formula({'[12]C': 2, 'H': 1, 'O': 1}), Formula({'[12]C': 1, '[13]C': 1, 'H': 1, 'O': 1}), "
        "Formula({'[13]C': 2, 'H': 1, 'O': 1})]"
    )
    isos = list(isotopologues(f, elements_with_isotopes=('C')))
    assert isos == check
    
    # Test with abundance reporting
    isos_with_abundance = list(isotopologues(f, report_abundance=True))
    assert all(len(x) == 2 for x in isos_with_abundance)

    # Test with abundance threshold
    check = eval(
        "[Formula({'[12]C': 2, '[1]H': 1, '[16]O': 1}), Formula({'[12]C': 1, '[13]C': 1, '[1]H': 1, '[16]O': 1}), "
        "Formula({'[13]C': 2, '[1]H': 1, '[16]O': 1})]"
    )
    isos = list(isotopologues(f, isotope_threshold=0.01))
    assert isos == check
    
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