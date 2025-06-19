# Simulation Based Metabolic Flux Inference (SBMFI)

SBMFI is a Python package for advanced simulation-based metabolic flux inference, with a focus on stable isotope labeling experiments (such as 13C-MFA). It provides a flexible framework for constructing, simulating, and analyzing atom-resolved metabolic models, and for inferring intracellular fluxes from experimental data.

**Key features:**
- Atom-resolved metabolic modeling and simulation
- Support for 13C and other isotope labeling experiments
- Probabilistic and simulation-based inference of metabolic fluxes
- Integration with LC-MS data, including isotope correction and adduct handling
- Tools for model construction, atom mapping, and custom metabolite definitions
- Extensible for systems biology, metabolic engineering, and isotope tracing studies

SBMFI is designed for researchers in systems biology, metabolic engineering, and related fields who need to analyze metabolic fluxes using isotope labeling data and advanced computational methods.

## Installation

From `gitlab` for now, need to churn it into a package at some point.

## License
[MIT](https://choosealicense.com/licenses/mit/)


### Tree structure
`tree /F /A | findstr /R /C:"^[A-Z]:\\" /C:"^[ |]*[+\\]-[^.]*$" /C:"\.py$"`

from sbmfi.core.model import model_builder_from_dict
from sbmfi.core.reaction import LabellingReaction

# Define your reactions and metabolites as dictionaries
reaction_kwargs = {...}
metabolite_kwargs = {...}

model = model_builder_from_dict(reaction_kwargs, metabolite_kwargs)

reaction = LabellingReaction(model.reactions.get_by_id('r1'))
atom_map, is_biomass = reaction.build_atom_map_from_string('A/ab + B/cd --> Q/acdb')
