import argparse
from pathlib import Path
import pickle

from cobra.io import read_sbml_model, write_sbml_model

import pta
from pta.concentrations_prior import ConcentrationsPrior
from pta.distributions import distribution_to_string, distribution_from_string
from pta.commons import Q
from pta.constants import default_confidence_level

def widen_concentrations_prior(concentrations_prior: ConcentrationsPrior, alpha=2.0):
    if alpha == 1.0:
        return concentrations_prior
    modified_distributions = {}
    for metabolite_info, distribution in concentrations_prior.metabolite_distributions.items():
        dist_str = distribution_to_string(distribution)
        name, p1, p2 = dist_str.split('|')
        p1, p2 = float(p1), float(p2)
        if 'Normal' in name:
            p2 *= alpha
        else:
            raise NotImplementedError
        modified_distributions[metabolite_info] = distribution_from_string(f'{name}|{p1}|{p2}')
    return ConcentrationsPrior(
        compound_cache=concentrations_prior._ccache,
        metabolite_distributions=modified_distributions,
        compartment_distributions=concentrations_prior._compartment_distributions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_pta_functions',
        description='executes various pta functions',
    )
    parser.add_argument('-f', dest='sbml_file', type=str)
    parser.add_argument('-v', action='count', default=0)  # verbosity level

    parser.add_argument('-prep', dest='prepta', type=int)
    parser.add_argument('-biomass_id', dest='biomass_id', type=str, default='BIOMASS_Ecoli_core_w_GAM')
    parser.add_argument('-atpm_id', dest='atpm_id', type=str, default='ATPM')

    parser.add_argument('-T', dest='T', type=float, default=310.15)  # needs to be in Kelvin
    parser.add_argument('-conc_prior', dest='conc_prior', type=str)
    parser.add_argument('-conc_alpha', dest='conc_alpha', type=float)

    parser.add_argument('-conf', dest='conf', type=float, default=default_confidence_level)

    args = parser.parse_args()

    sbml_file = args.sbml_file
    file_name = Path(sbml_file).stem
    file_prepta = f'{file_name}_prepta.xml'

    prepta = args.prepta
    if prepta:
        biomass_id = args.biomass_id
        atpm_id = args.atpm_id
        model = read_sbml_model(sbml_file)
        sa = pta.StructuralAssessment(model, biomass_id=biomass_id, atpm_id=atpm_id)
        pta.prepare_for_pta(model, biomass_id=biomass_id, atpm_id=atpm_id)
        write_sbml_model(model, filename=file_prepta)
    else:
        model = read_sbml_model(file_prepta)

    fname = f'{file_name}_tfs.p'

    conc_prior = args.conc_prior
    if conc_prior:
        fname = f'{file_name}_{conc_prior}_tfs.p'
        aero_prior = None
        if conc_prior.endswith('aero'):
            aero_prior = ConcentrationsPrior.load('M9_aerobic')

        if conc_prior.startswith('gluc'):
            concentrations_prior = ConcentrationsPrior.load('ecoli_M9_glc')
        else:
            raise NotImplementedError
        if aero_prior is not None:
            concentrations_prior.add(aero_prior)
    else:
        concentrations_prior = ConcentrationsPrior()

    alpha = args.conc_alpha
    if alpha:
        concentrations_prior = widen_concentrations_prior(concentrations_prior, alpha=alpha)

    flux_space = pta.FluxSpace.from_cobrapy_model(model=model)
    compartment_parameters = pta.CompartmentParameters.load("e_coli")
    compartment_parameters._T = Q(args.T, "K")  # NB this is to set the temperature to 37.0 oC
    thermo_space = pta.ThermodynamicSpace.from_cobrapy_model(
        model=model,
        parameters=compartment_parameters,
        concentrations=concentrations_prior,
        # constrained_rxns=constrained_rxns
    )

    problem = pta.PmoProblem(model, thermo_space, confidence_level=args.conf, solver="GUROBI")

    verbose = args.v
    status = problem.solve(verbose=verbose)
    if verbose > 0:
        print(f'PmoProblem solution status: {status}')
        assessment = pta.QuantitativeAssessment(problem)
        print(assessment.summary())
    if status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError('no thermodynamic space solutions, relax confidence levels, '
                         'concentration prior or compartment parameters')

    tfs_model = pta.TFSModel(flux_space, thermo_space, confidence_level=args.conf, solver="GUROBI")
    tfs_model.get_initial_points(num_points=1)  # NB making sure this works
    pickle.dump(tfs_model, open(fname, 'wb'))