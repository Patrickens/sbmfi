import pytest
import numpy as np
import pandas as pd
from cobra import Model, Reaction, Metabolite
from sbmfi.core.model import LabellingModel, EMU_Model, create_full_metabolite_kwargs, model_builder_from_dict
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
                'atom_map_str': 'A/ab + J --> P/ab', # has a co-factor
                'upper_bound': 100.0,
                'lower_bound': 0.0
            },
            'r2': {
                'atom_map_str': 'A/ab + B/cd --> Q/acdb',
                'upper_bound': 100.0,
                'lower_bound': 0.0
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
                'P': {'formula': 'C2H3O2', 'compartment': 'c', 'charge': -1},
                'J': {'formula': 'C1H1O1', 'compartment': 'c', 'charge': 0},  # metabolite without formula
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
            self, request, basic_reaction_kwargs, basic_metabolite_kwargs, 
            reaction_ids, expected, add_cofactors, only_labmets
    ):
        """Test various metabolite kwargs creation scenarios with different parameter combinations"""
        # Select relevant reactions and metabolites
        if isinstance(reaction_ids, list):
            reaction_kwargs = {rid: basic_reaction_kwargs[rid] for rid in reaction_ids}
        else:
            reaction_kwargs = reaction_ids  # Use custom reaction kwargs directly
            
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

    @pytest.mark.parametrize("reaction_kwargs,metabolite_ids,expected_error", [
        # Invalid atom mapping - too many atoms
        pytest.param(
            {'r1': {'atom_map_str': 'A/ab --> P/abc'}},
            ['A', 'P'],
            "Formula mismatch for metabolite P: formula has 2 carbon atoms but atom mapping has 3 atoms",
            id="too_many_atoms"
        ),
        # Invalid atom mapping - too few atoms
        pytest.param(
            {'r1': {'atom_map_str': 'A/ab --> P/a'}},
            ['A', 'P'],
            "Formula mismatch for metabolite P: formula has 2 carbon atoms but atom mapping has 1 atoms",
            id="too_few_atoms"
        )
    ])
    def test_create_full_metabolite_kwargs_errors(
            self, request, basic_metabolite_kwargs, reaction_kwargs, metabolite_ids, expected_error
    ):
        """Test error handling in metabolite kwargs creation"""
        metabolite_kwargs = {mid: basic_metabolite_kwargs[mid] for mid in metabolite_ids}

        with pytest.raises(ValueError, match=expected_error):
            create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs)

    @pytest.mark.parametrize("reaction_ids,metabolite_ids,expected", [
        # Basic model with single reaction
        pytest.param(
            ['r1'],
            ['A', 'P'],
            {'reactions': 1, 'metabolites': 2},
            id="single_reaction"
        ),
        # Model with multiple reactions
        pytest.param(
            ['r1', 'r2'],
            ['A', 'B', 'P', 'Q'],
            {'reactions': 2, 'metabolites': 4},
            id="multiple_reactions"
        ),
        # Model with biomass reaction
        pytest.param(
            ['r1', 'biomass'],
            ['A', 'P'],
            {'reactions': 2, 'metabolites': 2},
            id="biomass_reaction"
        )
    ])
    def test_model_creation_variants(
            self, request, basic_reaction_kwargs, basic_metabolite_kwargs, reaction_ids, metabolite_ids, expected
    ):
        """Test various model creation scenarios"""
        # Select relevant reactions and metabolites
        if isinstance(reaction_ids, list):
            reaction_kwargs = {rid: basic_reaction_kwargs[rid] for rid in reaction_ids if rid != 'biomass'}
            if 'biomass' in reaction_ids:
                reaction_kwargs['biomass'] = {'reaction_str': '0.1 A + 0.6 P --> '}
        else:
            reaction_kwargs = reaction_ids  # Use custom reaction kwargs directly
            
        metabolite_kwargs = {mid: basic_metabolite_kwargs[mid] for mid in metabolite_ids}

        model = model_builder_from_dict(reaction_kwargs, metabolite_kwargs)
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
        
        reaction_kwargs = {'r1': basic_reaction_kwargs['r1']}
        model.add_labelling_kwargs(reaction_kwargs, basic_metabolite_kwargs)
        
        # Verify reaction atom mapping
        r1 = model.reactions.get_by_id('r1')
        assert hasattr(r1, 'atom_map')
        assert len(r1.atom_map) == 2
        met_a = model.metabolites.get_by_id('A')
        met_p = model.metabolites.get_by_id('P')
        assert r1.atom_map[met_a][0] == -1
        assert np.array_equal(r1.atom_map[met_a][1], np.array([('a', 'b')]))
        assert r1.atom_map[met_p][0] == 1
        assert np.array_equal(r1.atom_map[met_p][1], np.array([('a', 'b')]))
        
        # Verify that all other reactions are cobra.Reaction objects
        for reaction in model.reactions:
            if reaction.id != 'r1':
                assert isinstance(reaction, Reaction)
                assert not isinstance(reaction, LabellingReaction)
        # Verify that all other metabolites are cobra.Metabolite objects
        for metabolite in model.metabolites:
            if metabolite.id not in ['A', 'P']:
                assert isinstance(metabolite, Metabolite)
                assert not isinstance(metabolite, LabelledMetabolite)
        
        # Verify model type attributes
        assert hasattr(model, '_TYPE_REACTION')
        assert model._TYPE_REACTION._TYPE_METABOLITE == LabelledMetabolite
        assert model._TYPE_REACTION == LabellingReaction