import re
from math import factorial
import numpy as np
from collections import defaultdict, Counter
from itertools import product, combinations_with_replacement
import math
import json
from pathlib import Path

# Get the directory containing this file
current_dir = Path(__file__).parent

# 1) Define a hook to convert any dict whose values are 2-element lists into
#    a dict with int keys and tuple values:
def isotope_hook(d):
    # detect an “isotopes” dict by checking that every value is a length-2 list
    if d and all(isinstance(v, list) and len(v) == 2 for v in d.values()):
        return {int(k): tuple(v) for k, v in d.items()}
    return d

# 2) Load back, using our hook for the inner dicts:
with open(current_dir / 'nist_mass.json', 'r') as f:
    _nist_mass = json.load(f, object_hook=isotope_hook)

# ALMOST ALL STOLEN FROM PYTEOMICS!

# unique _chnops ordering such that Formula.to_chnops() produces unique strings for every possible isotopomer
_chnops = ['C', 'H', 'N', 'O', 'P', 'S']
_chnops = _chnops + [element for element in _nist_mass.keys() if element not in _chnops]
_chnops = {k: v for k, v in zip(_chnops, range(len(_chnops)))}
_isotope_pat = r'(?:\[(\d+)\])?([A-Z][a-z]*)(\d*)'
_isotope_rex = re.compile(_isotope_pat)
_charge_pat = r'([+-])(\d*)$'
_charge_rex = re.compile(_charge_pat)
_formula_pat = fr'^({_isotope_pat})*({_charge_pat})*$'
_formula_rex = re.compile(_formula_pat.replace(r'\\', r'\''))


class FormulAlgebra(defaultdict, Counter):
    """A generic dictionary for compositions.
    Keys should be strings, values should be integers.
    Allows simple arithmetics."""

    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, int)
        Counter.__init__(self, *args, **kwargs)
        for k, v in list(self.items()):
            if not v:
                del self[k]

    def __str__(self):
        return '{}({})'.format(type(self).__name__, dict.__repr__(self))

    def __repr__(self):
        return str(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:  # should never happen
            p.text('{} object with a cyclic reference'.format(type(self).__name__))
        p.text(str(self))

    def __add__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] += cnt
        return result

    def __iadd__(self, other):
        for elem, cnt in other.items():
            self[elem] += cnt
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] -= cnt
        return result

    def __isub__(self, other):
        for elem, cnt in other.items():
            self[elem] -= cnt
        return self

    def __rsub__(self, other):
        return (self - other) * (-1)

    def __mul__(self, other):
        if not isinstance(other, int):
            raise ValueError('Cannot multiply Composition by non-integer',
                                 other)
        return type(self)({k: v * other for k, v in self.items()})

    def __imul__(self, other):
        if not isinstance(other, int):
            raise ValueError('Cannot multiply Composition by non-integer',
                                 other)
        for elem in self:
            self[elem] *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        self_items = {i for i in self.items() if i[1]}
        other_items = {i for i in other.items() if i[1]}
        return self_items == other_items

    # override default behavior:
    # we don't want to add 0's to the dictionary
    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        if math.isnan(value):
            value = 0
        if isinstance(value, float):
            value = int(round(value))
        elif not isinstance(value, int):
            raise ValueError('Only integers allowed as values in '
                                 'Composition, got {}.'.format(type(value).__name__))
        if value:  # reject 0's
            super(FormulAlgebra, self).__setitem__(key, value)
        elif key in self:
            del self[key]

    def copy(self):
        return type(self)(self)

    def __copy__(self):
        return self.copy()

    def __reduce__(self):
        class_, args, state, list_iterator, dict_iterator = super(
            FormulAlgebra, self).__reduce__()
        # Override the reduce of defaultdict so we do not provide the
        # `int` type as the first argument
        # which prevents from correctly unpickling the object
        args = ()
        return class_, args, state, list_iterator, dict_iterator


class Formula(FormulAlgebra):
    """Chemical formula stored as a dictionary-like structure.
    Contains convenient functions like the calculation of mono-isotopic mass and
    natural abundance.
    """
    # NOTE: I gave formula a default empty string, and hopefully this fixes the pickling issue in multiprocessing
    def __init__(self, formula='', charge=0):
        if formula is None:
            formula = ''
        defaultdict.__init__(self, int)
        if charge:
            self['-'] = -charge
        if isinstance(formula, dict):
            # NOTE: assume that if we are copying, that the incoming Formula is already properly formatted
            for isotop, number in formula.items():
                k, elem = self._parse_isotope_string(isotop)
                if not elem in _nist_mass:
                    raise ValueError('Unknown chemical element: ' + elem)
                self[self._make_isotope_string(k, elem)] += number
        elif isinstance(formula, str):
            # NOTE: for formulas with negative numbers like neutral loss H2O,
            #  the number and charge are confused; 'H-2O-1-0' works since its converted to 'H-2O-1'
            if _formula_rex.match(formula) is None:
                raise ValueError(f'Not a properly formatted formula: {formula}')

            charge = _charge_rex.findall(formula)
            if charge:
                polarity, number = charge[0]
                z = 1
                if number:
                    z = int(number)
                if polarity == '+':
                    z = -z
                self['-'] += z

            for isotope, elem, number in _isotope_rex.findall(formula):
                if not elem in _nist_mass:
                    raise ValueError('Unknown chemical element: ' + elem)
                self[self._make_isotope_string(int(isotope) if isotope else 0, elem)] += \
                    int(number) if number else 1

    @staticmethod
    def _parse_isotope_string(isotope_string: str):
        if isotope_string == '-':
            return 0, isotope_string
        num, elem, _ = _isotope_rex.match(isotope_string).groups()
        k = int(num) if num else 0
        if (elem not in _nist_mass) or (k not in _nist_mass[elem]):
            raise ValueError(f'Not a chemical element G')
        return k, elem

    @staticmethod
    def _make_isotope_string(k: int, elem: str):
        """Form a string label for an isotope."""
        if (elem not in _nist_mass) or (k not in _nist_mass[elem]):
            raise ValueError(f'Conjo papi')

        if k == 0:
            return elem
        else:
            return f'[{k}]{elem}'

    def mass(self, charge=0, average=False, ion=True, abundances=_nist_mass):
        """ calculates the mass of the formula object
        TODO: TEST ion=False/True!!
        Parameters
        ----------
        charge : int, default 0
            Adds or subtracts `charge` electrons from the mass and divides by charge.
            Used for m/z calculations in mass-spectrometry applications.
        average : bool, default False
            When True, calculates the average mass. Masses of all isotope of the Formulas
            atoms are multiplied by their abundance
        """
        # Calculate mass
        mass = 0.0
        for isotope_string, amount in self.items():
            if isotope_string == '-':
                amount -= charge

            k, elem = self._parse_isotope_string(isotope_string)
            # Calculate average mass if required and the isotope number is
            # not specified.
            if (not k) and average:
                for isotope, data in abundances[elem].items():
                    if isotope:
                        mass += (amount * data[0] * data[1])
            else:
                mass += (amount * abundances[elem][k][0])

        # Calculate m/z if required
        nE = charge - self['-']
        if (nE != 0) and ion:
            mass = abs(mass/nE)
        elif ion:
            raise ValueError('Not a charged molecule!')
        return mass

    def to_chnops(self) -> str:
        """Unique string representation of a given formula; elements
        C,H,N,O,P and S are at the start of the string
        """
        charge = self.pop('-', 0)
        sorter = np.zeros((2, len(self)), dtype=np.int64)
        for i, (isotope_string, number) in enumerate(self.items()):
            k, elem = self._parse_isotope_string(isotope_string)
            sorter[1,i] = _chnops[elem]
            sorter[0,i] = k
        to_sort = np.array([k + (str(v) if v != 1 else '') for k, v in self.items()])
        indices = np.lexsort(keys=sorter)
        the_string = ''.join(to_sort[indices])
        if charge > 0:
            the_string += '-' if charge == 1 else f'-{charge}'
        elif charge < 0:
            the_string += '+' if charge == -1 else f'+{-charge}'
        self['-'] = charge
        return the_string

    def add_C13(self, nC13:int):
        """Exchanges C[12] for C[13] atoms in a Formula"""
        if nC13 > 0:
            self['[13]C'] = nC13
            self['C'] -= nC13
        elif nC13 < 0:
            self['C'] -= nC13
            self['[13]C'] += nC13

    def full_isotope(self):
        """Returns a Formula with all atoms fully isotopically labeled"""
        result = self.copy()
        for isotope_string, amount in self.items():
            if isotope_string == '-':
                continue
            k, elem = self._parse_isotope_string(isotope_string)
            if k == 0:
                # Find the most abundant isotope
                max_abundance = 0
                max_isotope = 0
                for isotope, data in _nist_mass[elem].items():
                    if data[1] > max_abundance:
                        max_abundance = data[1]
                        max_isotope = isotope
                result[self._make_isotope_string(max_isotope, elem)] = amount
                del result[isotope_string]
        return result

    def no_isotope(self):
        """Returns a Formula with all atoms unlabeled"""
        result = self.copy()
        for isotope_string, amount in self.items():
            if isotope_string == '-':
                continue
            k, elem = self._parse_isotope_string(isotope_string)
            if k != 0:
                result[elem] = amount
                del result[isotope_string]
        return result

    def abundance(self, abundances=_nist_mass):
        """Calculates the natural abundance of the formula"""
        abundance = 1.0
        for isotope_string, amount in self.items():
            if isotope_string == '-':
                continue
            k, elem = self._parse_isotope_string(isotope_string)
            if k == 0:
                # For unlabeled atoms, use the natural abundance
                for isotope, data in abundances[elem].items():
                    if isotope:
                        abundance *= data[1] ** amount
            else:
                # For labeled atoms, use the specific isotope abundance
                abundance *= abundances[elem][k][1] ** amount
        return abundance

    def shift(self):
        """Returns the mass shift of the formula relative to the unlabeled formula"""
        return self.mass() - self.no_isotope().mass()


def isotopologues(
        formula,
        report_abundance=False,
        elements_with_isotopes=None,
        isotope_threshold=5e-4,
        overall_threshold=0.0,
        abundances=_nist_mass,
        n_mdv = None,
    ):
    """Generate all possible isotopologues of a formula.
    
    Parameters
    ----------
    formula : Formula
        The formula to generate isotopologues for
    report_abundance : bool, default False
        If True, also return the natural abundance of each isotopologue
    elements_with_isotopes : list, optional
        List of elements to consider for isotopic labeling. If None, all elements are considered.
    isotope_threshold : float, default 5e-4
        Minimum abundance for an isotope to be considered
    overall_threshold : float, default 0.0
        Minimum overall abundance for an isotopologue to be included
    abundances : dict, default _nist_mass
        Dictionary of isotope abundances
    n_mdv : int, optional
        If provided, only return isotopologues with this number of mass differences
        
    Returns
    -------
    list
        List of isotopologues (and their abundances if report_abundance is True)
    """
    if elements_with_isotopes is None:
        elements_with_isotopes = list(abundances.keys())
    
    # Get all possible isotopes for each element
    isotopes = {}
    for elem in elements_with_isotopes:
        if elem in abundances:
            isotopes[elem] = [k for k, v in abundances[elem].items() 
                            if v[1] > isotope_threshold]
    
    # Generate all possible combinations
    combinations = []
    for elem in elements_with_isotopes:
        if elem in formula:
            n = formula[elem]
            if n > 0:
                combinations.append([(elem, k) for k in isotopes.get(elem, [0])])
    
    # Generate all isotopologues
    result = []
    for combo in product(*combinations):
        new_formula = formula.copy()
        for elem, k in combo:
            if k != 0:
                new_formula[f'[{k}]{elem}'] = new_formula.pop(elem)
        
        if n_mdv is not None:
            if new_formula.shift() != n_mdv:
                continue
                
        if report_abundance:
            abundance = new_formula.abundance(abundances)
            if abundance > overall_threshold:
                result.append((new_formula, abundance))
        else:
            result.append(new_formula)
    
    return result 