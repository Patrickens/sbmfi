import pytest
import numpy as np
import pandas as pd
from cobra import Model, Reaction, Metabolite
from sbmfi.core.model import (
    LabellingModel,
    EMU_Model,
    create_full_metabolite_kwargs,
    model_builder_from_dict,
    process_biomass_reaction
)
from sbmfi.core.linalg import LinAlg
from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU, IsoCumo
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction


class TestModelBuilder:
    """Test suite for model building functions"""

    @pytest.fixture
    def basic_reaction_kwargs(self):
        return {
            'a_in': {
                'atom_map_str': '--> A/ab',  # exchange reaction
                'upper_bound': 100.0,
                'lower_bound': 0.0
            },
            'r1': {
                'atom_map_str': 'A/ab + J --> P/ab',  # has a co-factor
                'upper_bound': 100.0,
                'lower_bound': 0.0
            },
            'r2': {
                'atom_map_str': 'A/ab + B/cd --> Q/acdb',  # default bounds
            },
            'r3': {
                'atom_map_str': 'Q/acdb --> R/cd + S/ba',
                'upper_bound': 100.0,
                'lower_bound': 0.0
            },
            'bm': {
                'atom_map_str': 'biomass --> ',  # biomass reaction
                'reaction_str': '0.3H + 0.6P + 0.5B + 0.1Q --> ∅',
                'upper_bound': 100.0,
                'lower_bound': 0.0
            },
            'rp': {
                'atom_map_str': 'A/ab + B/cd + A/ef --> L/acf',  # pseudo-reaction
                'pseudo': True
            },
        }

    @pytest.fixture
    def basic_metabolite_kwargs(self):
        return {
            'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
            'B': {'formula': 'C2H6O', 'compartment': 'c', 'charge': 0},
            'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
            # 'Q': {},  # to test a metabolite without annotations
            'R': {}, # metabolite without formula
            'S': {'formula': 'C2H4O3', 'compartment': 'p', 'charge': -1},
            'J': {'formula': 'C1H1O1', 'compartment': 'c', 'charge': 0}  # cofactor
        }

    @pytest.mark.parametrize("reaction_ids,expected,add_cofactors,only_labmets", [
        # Basic model creation
        pytest.param(
            ['r1', 'r3'],
            {
                'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
                'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
                'Q': {'formula': 'C4'},
                'R': {'formula': 'C2'},
                'S': {'formula': 'C2H4O3', 'compartment': 'p', 'charge': -1},
            },
            False,
            False,
            id="two_reactions_no_cofactors"
        ),
        # Multiple reactions, biomass and co-factor=True
        pytest.param(
            ['r1', 'bm'],
            {
                'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
                'J': {'formula': 'C1H1O1', 'compartment': 'c', 'charge': 0},
                'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
                'H': {'compartment': 'c'},
                'B': {'compartment': 'c'},
                'Q': {'compartment': 'c'}
            },
            True,
            False,
            id="multiple_reactions_biomass_cofactors"
        ),
        # Test only labelled metabolites
        pytest.param(
            ['r1'],
            {
                'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
                'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
            },
            False,
            True,
            id="only_labelled"
        ),
        # Test with cofactors and only labelled metabolites
        pytest.param(
            ['r1'],
            {
                'A': {'formula': 'C2H4O2', 'compartment': 'c', 'charge': -1},
                'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
            },
            True,
            True,
            id="cofactors_and_labelled"
        )
    ])
    def test_create_full_metabolite_kwargs_variants(
            self, basic_reaction_kwargs, basic_metabolite_kwargs, 
            reaction_ids, expected, add_cofactors, only_labmets
    ):
        """Test various metabolite kwargs creation scenarios with different parameter combinations"""
        # Select relevant reactions and metabolites
        reaction_kwargs = {rid: basic_reaction_kwargs[rid] for rid in reaction_ids}

        result = create_full_metabolite_kwargs(
            reaction_kwargs, 
            basic_metabolite_kwargs,
            infer_formula=True,
            add_cofactors=add_cofactors,
            only_labmets=only_labmets
        )
        if not add_cofactors or only_labmets:
            assert 'J' not in result

        if only_labmets:
            for met_id in basic_metabolite_kwargs.keys():
                if met_id in ['A', 'P']:
                    continue
                assert met_id not in result

        for met_id, met_kwargs in expected.items():
            assert met_id in result
            result_kwargs = result[met_id]

            for kwd, val in met_kwargs.items():
                assert kwd in result_kwargs
                assert result_kwargs[kwd] == val

    def test_create_full_metabolite_kwargs_errors(
            self, basic_metabolite_kwargs
    ):
        """Test error handling in metabolite kwargs creation when atom mapping doesn't match formula"""
        reaction_kwargs = {'r1': {'atom_map_str': 'A/ab --> P/abc'}}
        expected_error = "Formula mismatch for metabolite P: formula has 2 carbon atoms but atom mapping has 3 atoms"

        with pytest.raises(ValueError, match=expected_error):
            create_full_metabolite_kwargs(reaction_kwargs, basic_metabolite_kwargs)

    @pytest.mark.parametrize("reaction_ids,expected", [
        # Basic model with single reaction
        pytest.param(
            ['r1'],
            {'reactions': 1, 'metabolites': 3},
            id="single_reaction"
        ),
        # Model with multiple reactions
        pytest.param(
            ['r1', 'r2'],
            {'reactions': 2, 'metabolites': 3 + 2},
            id="multiple_reactions"
        ),
        # Model with biomass reaction
        pytest.param(
            ['r1', 'bm'],
            {'reactions': 2, 'metabolites': 3 + 3},
            id="biomass_reaction"
        )
    ])
    def test_model_creation_variants(
            self, basic_reaction_kwargs, basic_metabolite_kwargs, reaction_ids, expected
    ):
        """Test various model creation scenarios"""
        # Select relevant reactions and metabolites
        reaction_kwargs = {rid: basic_reaction_kwargs[rid] for rid in reaction_ids}
        model = model_builder_from_dict(reaction_kwargs, basic_metabolite_kwargs)
        assert len(model.reactions) == expected['reactions']
        assert len(model.metabolites) == expected['metabolites']

    def test_add_labelling_kwargs(self, basic_reaction_kwargs, basic_metabolite_kwargs):
        """Test adding labelling information to a model"""
        # Create model with a single reaction for simplicity
        model = model_builder_from_dict(
            reaction_kwargs=basic_reaction_kwargs,
            metabolite_kwargs=basic_metabolite_kwargs
        )
        model = LabellingModel(LinAlg('numpy'), model)

        # Include all reactions except a_in
        reaction_kwargs = {rid: kwargs for rid, kwargs in basic_reaction_kwargs.items() if rid != 'a_in'}
        model.add_labelling_kwargs(reaction_kwargs, basic_metabolite_kwargs)

        # Verify reaction atom mapping for r1
        r1 = model.reactions.get_by_id('r1')
        assert hasattr(r1, 'atom_map')
        assert len(r1.atom_map) == 2
        met_a = model.metabolites.get_by_id('A')
        met_p = model.metabolites.get_by_id('P')
        assert r1.atom_map[met_a][0] == -1
        assert np.array_equal(r1.atom_map[met_a][1], np.array([('a', 'b')]))
        assert r1.atom_map[met_p][0] == 1
        assert np.array_equal(r1.atom_map[met_p][1], np.array([('a', 'b')]))

        # Verify reaction bounds for all reactions
        assert r1.lower_bound == 0.0  # from fixture
        assert r1.upper_bound == 100.0  # from fixture
        
        r2 = model.reactions.get_by_id('r2')
        assert r2.lower_bound == 0.0  # default
        assert r2.upper_bound == 1000.0  # default
        
        r3 = model.reactions.get_by_id('r3')
        assert r3.lower_bound == 0.0  # default
        assert r3.upper_bound == 100.0  # default
        
        bm = model.reactions.get_by_id('bm')
        assert bm.lower_bound == 0.0  # from fixture
        assert bm.upper_bound == 100.0  # from fixture

        # Verify pseudo reaction status
        rp = model.pseudo_reactions.get_by_id('rp')
        assert rp.pseudo is True
        for rid in ['r1', 'r2', 'r3', 'bm']:
            assert getattr(model.reactions.get_by_id(rid), 'pseudo', False) is False

        # Verify that all other reactions are cobra.Reaction objects
        for reaction in model.reactions:
            if reaction.id not in ['r1', 'r2', 'r3', 'bm', 'rp']:
                assert isinstance(reaction, Reaction)
                assert not isinstance(reaction, LabellingReaction)
            else:
                assert isinstance(reaction, LabellingReaction)
        # Verify that all other metabolites are cobra.Metabolite objects
        for metabolite in model.metabolites:
            if metabolite.id not in ['A', 'P', 'B', 'Q', 'R', 'S']:
                assert isinstance(metabolite, Metabolite)
                assert not isinstance(metabolite, LabelledMetabolite)
            else:
                assert isinstance(metabolite, LabelledMetabolite)

        # Verify model type attributes
        assert hasattr(model, '_TYPE_REACTION')
        assert model._TYPE_REACTION._TYPE_METABOLITE == LabelledMetabolite
        assert model._TYPE_REACTION == LabellingReaction

    def test_process_biomass_reaction(self):
        """Test processing biomass reaction from reaction kwargs with different scenarios"""
        # Test case 1: No biomass reaction
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/ab --> P/ab'},
            'r2': {'atom_map_str': 'B/cd --> Q/cd'}
        }
        assert process_biomass_reaction(reaction_kwargs) is None

        # Test case 2: Single biomass reaction with default name and coefficients
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/ab --> P/ab'},
            'biomass': {
                'atom_map_str': 'biomass --> ',
                'reaction_str': '0.3H + 0.6P + 0.5B + 0.1Q --> ∅'
            }
        }
        biomass_id, biomass_coeff = process_biomass_reaction(reaction_kwargs)
        assert biomass_id == 'biomass'
        assert biomass_coeff == {'H': -0.3, 'P': -0.6, 'B': -0.5, 'Q': -0.1}

        # Test case 3: Single biomass reaction with custom name
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/ab --> P/ab'},
            'growth': {
                'atom_map_str': 'biomass --> ',
                'reaction_str': '0.4H + 0.7P --> ∅'
            }
        }
        biomass_id, biomass_coeff = process_biomass_reaction(reaction_kwargs)
        assert biomass_id == 'growth'
        assert biomass_coeff == {'H': -0.4, 'P': -0.7}

        # Test case 4: Multiple biomass reactions (should raise error)
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/ab --> P/ab'},
            'biomass1': {'atom_map_str': 'biomass --> '},
            'biomass2': {'atom_map_str': 'biomass --> '}
        }
        with pytest.raises(ValueError, match="Multiple biomass reactions found"):
            process_biomass_reaction(reaction_kwargs)

        # Test case 5: Biomass reaction without reaction_str
        reaction_kwargs = {
            'r1': {'atom_map_str': 'A/ab --> P/ab'},
            'biomass': {'atom_map_str': 'biomass --> '}
        }
        biomass_id, biomass_coeff = process_biomass_reaction(reaction_kwargs)
        assert biomass_id == 'biomass'
        assert biomass_coeff == {}


class TestLabellingModelSubstrateLabelling:
    @pytest.fixture
    def model(self, basic_reaction_kwargs, basic_metabolite_kwargs):
        cobra_model = model_builder_from_dict(basic_reaction_kwargs, basic_metabolite_kwargs)
        model = LabellingModel(LinAlg('numpy'), cobra_model)
        # Add labelling info for all reactions except a_in (exchange)
        reaction_kwargs = {rid: kwargs for rid, kwargs in basic_reaction_kwargs.items() if rid != 'a_in'}
        model.add_labelling_kwargs(reaction_kwargs, basic_metabolite_kwargs)
        return model

    @pytest.fixture
    def valid_substrate_labelling(self):
        # A/00 and A/11 for metabolite A (2 carbons)
        return pd.Series({'A/00': 0.5, 'A/11': 0.5}, name='test_labelling')

    def test_set_and_get_substrate_labelling(self, model, valid_substrate_labelling):
        model.set_substrate_labelling(valid_substrate_labelling)
        # Check property returns the same values (rounded)
        out = model.substrate_labelling
        assert set(out.index) == set(valid_substrate_labelling.index)
        np.testing.assert_allclose(sorted(out.values), sorted(valid_substrate_labelling.values), atol=1e-4)
        # Check substrate_metabolites property
        mets = model.substrate_metabolites
        assert len(mets) == 1
        assert mets[0].id == 'A'

    def test_set_same_labelling_twice_idempotent(self, model, valid_substrate_labelling):
        model.set_substrate_labelling(valid_substrate_labelling)
        # Should not error or change on second call
        model.set_substrate_labelling(valid_substrate_labelling)
        out = model.substrate_labelling
        assert set(out.index) == set(valid_substrate_labelling.index)

    @pytest.mark.parametrize("labelling,errmsg", [
        (pd.Series({'A/00': -0.1, 'A/11': 1.1}, name='test_labelling'),
         'Negative or over 1 value in substrate labelling'),
        (pd.Series({'A/00': 0.3, 'A/11': 0.3}, name='test_labelling'),
         r'substrate labeling fractions of metabolite A do not sum up to 1.0'),
    ])
    def test_set_substrate_labelling_value_errors(self, model, labelling, errmsg):
        with pytest.raises(ValueError, match=errmsg):
            model.set_substrate_labelling(labelling)

    def test_illegally_reversible_substrate_reaction(self, model, valid_substrate_labelling):
        # Make the substrate reaction reversible (rho_max != 0.0)
        a_in = model.reactions.get_by_id('a_in')
        # Patch to LabellingReaction and set rho_max
        model.reactions._replace_on_id(LabellingReaction(a_in, rho_max=1.0))
        with pytest.raises(ValueError, match=r'substrate reaction is illegaly reversible'):
            model.set_substrate_labelling(valid_substrate_labelling)

    def test_substrate_reaction_zero_bounds(self, model, valid_substrate_labelling):
        # Set both bounds to zero
        a_in = model.reactions.get_by_id('a_in')
        model.reactions._replace_on_id(LabellingReaction(a_in, rho_max=0.0, lower_bound=0.0, upper_bound=0.0))
        with pytest.raises(ValueError, match=r'substrate reaction a_in for metabolite A has \(0, 0\) bounds'):
            model.set_substrate_labelling(valid_substrate_labelling)

    def test_no_substrate_reaction(self, model, valid_substrate_labelling):
        # Remove all reactions from A
        met = model.metabolites.get_by_id('A')
        met._reaction.clear()
        with pytest.raises(ValueError, match=r'metabolite A has no substrate reactions'):
            model.set_substrate_labelling(valid_substrate_labelling)


class TestEMUModelSubstrateLabelling:
    @pytest.fixture
    def emu_model(self, basic_reaction_kwargs, basic_metabolite_kwargs):
        cobra_model = model_builder_from_dict(basic_reaction_kwargs, basic_metabolite_kwargs)
        model = EMU_Model(LinAlg('numpy'), cobra_model)
        reaction_kwargs = {rid: kwargs for rid, kwargs in basic_reaction_kwargs.items() if rid != 'a_in'}
        model.add_labelling_kwargs(reaction_kwargs, basic_metabolite_kwargs)
        # Minimal setup for EMU state
        model._xemus = {2: []}
        model._yemus = {2: []}
        model._emu_indices = {}
        model._A_tot = {2: np.zeros((1, 0, 0))}
        model._B_tot = {2: np.zeros((1, 0, 0))}
        model._X = {2: np.zeros((1, 0, 3))}
        model._Y = {2: np.zeros((1, 0, 3))}
        model._dXdv = {2: np.zeros((1, 0, 3))}
        model._dYdv = {2: np.zeros((1, 0, 3))}
        return model

    @pytest.fixture
    def valid_substrate_labelling(self):
        return pd.Series({'A/00': 0.5, 'A/11': 0.5}, name='emu_labelling')

    def test_set_and_get_substrate_labelling(self, emu_model, valid_substrate_labelling):
        emu_model.set_substrate_labelling(valid_substrate_labelling)
        out = emu_model.substrate_labelling
        assert set(out.index) == set(valid_substrate_labelling.index)
        np.testing.assert_allclose(sorted(out.values), sorted(valid_substrate_labelling.values), atol=1e-4)

    def test_set_same_labelling_twice_idempotent(self, emu_model, valid_substrate_labelling):
        emu_model.set_substrate_labelling(valid_substrate_labelling)
        # Should use cache, not error
        emu_model.set_substrate_labelling(valid_substrate_labelling)
        out = emu_model.substrate_labelling
        assert set(out.index) == set(valid_substrate_labelling.index)

    @pytest.mark.parametrize("labelling,errmsg", [
        (pd.Series({'A/00': -0.1, 'A/11': 1.1}, name='emu_labelling'),
         'Negative or over 1 value in substrate labelling'),
        (pd.Series({'A/00': 0.3, 'A/11': 0.3}, name='emu_labelling'),
         r'substrate labeling fractions of metabolite A do not sum up to 1.0'),
    ])
    def test_set_substrate_labelling_value_errors(self, emu_model, labelling, errmsg):
        with pytest.raises(ValueError, match=errmsg):
            emu_model.set_substrate_labelling(labelling)

    def test_illegally_reversible_substrate_reaction(self, emu_model, valid_substrate_labelling):
        a_in = emu_model.reactions.get_by_id('a_in')
        emu_model.reactions._replace_on_id(EMU_Reaction(a_in, rho_max=1.0))
        with pytest.raises(ValueError, match=r'substrate reaction is illegaly reversible'):
            emu_model.set_substrate_labelling(valid_substrate_labelling)

    def test_substrate_reaction_zero_bounds(self, emu_model, valid_substrate_labelling):
        a_in = emu_model.reactions.get_by_id('a_in')
        emu_model.reactions._replace_on_id(EMU_Reaction(a_in, rho_max=0.0, lower_bound=0.0, upper_bound=0.0))
        with pytest.raises(ValueError, match=r'substrate reaction a_in for metabolite A has \(0, 0\) bounds'):
            emu_model.set_substrate_labelling(valid_substrate_labelling)

    def test_no_substrate_reaction(self, emu_model, valid_substrate_labelling):
        met = emu_model.metabolites.get_by_id('A')
        met._reaction.clear()
        with pytest.raises(ValueError, match=r'metabolite A has no substrate reactions'):
            emu_model.set_substrate_labelling(valid_substrate_labelling)

    def test_empty_Y_cache_raises(self, emu_model, valid_substrate_labelling):
        # Simulate a cache hit but with empty _Y
        emu_model._labelling_repo['emu_labelling'] = {'_substrate_labelling': {}}
        emu_model._Y = {}
        emu_model._yemus = {2: []}
        with pytest.raises(ValueError):
            emu_model.set_substrate_labelling(valid_substrate_labelling)