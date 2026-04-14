import os
import inspect
from pathlib import Path

### Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
BASE_DIR   = os.path.join(*Path(SCRIPT_DIR).parts[:-2])
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'models')
SIM_DIR    = os.path.join(BASE_DIR, 'simulations')

### Solver configuration
# Override via environment variables: SBMFI_COBRA_SOLVER, SBMFI_CVXPY_SOLVER
COBRA_SOLVER = os.environ.get('SBMFI_COBRA_SOLVER', 'glpk')
CVXPY_SOLVER = os.environ.get('SBMFI_CVXPY_SOLVER', 'CLARABEL')

# Apply cobra solver globally so all Model() constructions use the right backend.
# Must happen before any cobra.Model() is instantiated.
import cobra.core.configuration as _cobra_cfg
_cobra_cfg.Configuration().solver = COBRA_SOLVER


if __name__ == "__main__":
    print(BASE_DIR)
    print(MODEL_DIR)
    print(f'cobra solver: {COBRA_SOLVER}')
    print(f'cvxpy solver: {CVXPY_SOLVER}')
