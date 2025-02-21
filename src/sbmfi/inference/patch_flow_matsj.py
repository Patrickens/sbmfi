from typing import Callable, Optional, Sequence, Tuple, Union
from torchdiffeq import odeint
from tqdm import tqdm
from flow_matching.solver.riemannian_ode_solver import (
    RiemannianODESolver, interp, _euler_step, _midpoint_step, _rk4_step
)
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils.manifolds import Manifold
from flow_matching.utils import gradient, ModelWrapper
import torch
from torch import Tensor
import math
from sbmfi.core.linalg import LinAlg
from torch import nn
from manifolds import BallManifold, ConvexPolytopeManifold


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, t=t)


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(self, input_dim, time_dim: int = 1, hidden_dim: int = 128, hidden_layers=6):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), Swish()]

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            *layers,
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1).float()
        h = torch.cat([x, t], dim=1).float()
        output = self.main(h)

        return output.reshape(*sz)

def sample_and_div(
        ode_solver: ODESolver,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        return_div: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
) -> Union[Tensor, Sequence[Tensor], Tuple[Tensor, Tensor]]:
    """Sample forward and optionally compute log probabilities.

    Args:
        return_div: If True, compute and return log probabilities. Requires log_p0.
        log_p0: Log probability function of initial distribution, required if return_log_p=True
    """
    if return_div:
        if not exact_divergence:
            z = (torch.randn_like(x_init).to(x_init.device) < 0) * 2.0 - 1.0

    def ode_func(t, x):
        return ode_solver.velocity_model(x=x, t=t, **model_extras)

    def dynamics_func(t, states):
        xt = states[0]
        with torch.set_grad_enabled(True):
            xt.requires_grad_()
            ut = ode_func(t, xt)

            if exact_divergence:
                # Compute exact divergence
                div = 0
                for i in range(ut.flatten(1).shape[1]):
                    div += gradient(ut[:, i], xt, create_graph=True)[:, i]
            else:
                # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                ut_dot_z = torch.einsum(
                    "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt)
                div = torch.einsum(
                    "ij,ij->i",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

        return ut.detach(), div.detach()

    ode_opts = {"step_size": step_size} if step_size is not None else {}

    if return_div:
        y_init = (x_init, torch.zeros(x_init.shape[0], device=x_init.device))
        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )
        if return_intermediates:
            return sol, log_det[-1]
        else:
            return sol[-1], log_det[-1, None]

    sol = odeint(
        ode_func,
        x_init,
        time_grid,
        method=method,
        options=ode_opts,
        atol=atol,
        rtol=rtol,
    )
    if return_intermediates:
        return sol
    else:
        return sol[-1]


def riem_sample_and_div_old(
        riem_solver: RiemannianODESolver,
        x_init: Tensor,
        step_size: float,
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        return_log_p: bool = False,
        exact_divergence: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
) -> Union[Tensor, Sequence[Tensor], Tuple[Tensor, Tensor]]:
    """Sample forward on the manifold and optionally compute log probabilities.

    Args:
        return_log_p: If True, compute and return log probabilities. Requires log_p0.
        log_p0: Log probability function of initial distribution, required if return_log_p=True
    """
    if return_log_p:
        if not exact_divergence:
            z = (torch.randn_like(x_init).to(x_init.device) < 0) * 2.0 - 1.0
            z = riem_solver.manifold.proju(x_init, z) if proju else z

    step_fns = {
        "euler": _euler_step,
        "midpoint": _midpoint_step,
        "rk4": _rk4_step,
    }
    assert method in step_fns.keys(), f"Unknown method {method}"
    step_fn = step_fns[method]

    # Prepare time discretization
    time_grid = torch.sort(time_grid.to(device=x_init.device)).values
    if step_size is None:
        t_discretization = time_grid
        n_steps = len(time_grid) - 1
    else:
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        assert (
                       t_final - t_init
               ) > step_size, f"Time interval [min(time_grid), max(time_grid)] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = math.ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )

    t0s = t_discretization[:-1]

    if verbose:
        t0s = tqdm(t0s)

    if return_intermediates:
        path = []
        i_ret = 0
        if return_log_p:
            log_det = []

    if return_intermediates:
        path = []
        i_ret = 0

    with torch.set_grad_enabled(enable_grad):
        xt = x_init
        if return_log_p:
            accum_div = torch.zeros(x_init.shape[0], device=x_init.device)

        for t0, t1 in zip(t0s, t_discretization[1:]):
            dt = t1 - t0

            # Compute velocity and divergence if needed
            if return_log_p:
                with torch.set_grad_enabled(True):
                    xt.requires_grad_()
                    ut = riem_solver.velocity_model(x=xt, t=t0, **model_extras)
                    if proju:
                        ut = riem_solver.manifold.proju(xt, ut)

                    if exact_divergence:
                        div = 0
                        for i in range(ut.flatten(1).shape[1]):
                            ei = torch.zeros_like(ut)
                            ei[:, i] = 1
                            if proju:
                                ei = riem_solver.manifold.proju(xt, ei)
                            div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                            christoffel = riem_solver.manifold.christoffel(xt, ei, ut) if hasattr(riem_solver.manifold, 'christoffel') else 0
                            div += div_i - christoffel
                    else:
                        u_dot_z = riem_solver.manifold.inner(xt, ut, z)
                        grad_u_dot_z = gradient(u_dot_z, xt)
                        if proju:
                            grad_u_dot_z = riem_solver.manifold.proju(xt, grad_u_dot_z)
                        div = riem_solver.manifold.inner(xt, grad_u_dot_z, z)

                    accum_div = accum_div + div * dt
                    ut = ut.detach()
            else:
                ut = riem_solver.velocity_model(x=xt, t=t0, **model_extras)
                if proju:
                    ut = riem_solver.manifold.proju(xt, ut)

            # Take integration step
            xt_next = step_fn(
                lambda x, t: ut,
                xt,
                t0,
                dt,
                manifold=riem_solver.manifold,
                projx=projx,
                proju=proju,
            )

            if return_intermediates:
                while (
                        i_ret < len(time_grid)
                        and t0 <= time_grid[i_ret]
                        and time_grid[i_ret] <= t1
                ):
                    interp_x = interp(
                        riem_solver.manifold, xt, xt_next, t0, t1, time_grid[i_ret]
                    )
                    path.append(interp_x)
                    if return_log_p:
                        log_det.append(accum_div)
                    i_ret += 1

            xt = xt_next

    if return_intermediates:
        if return_log_p:
            return torch.stack(path, dim=0), torch.stack(log_det, dim=0)[-1]
        return torch.stack(path, dim=0)
    else:
        if return_log_p:
            return xt, accum_div
        return xt


def riem_sample_and_div(
    self,
    x_init: Tensor,
    step_size: float,
    return_div: bool = True,
    projx: bool = True,
    proju: bool = True,
    method: str = "euler",
    time_grid: Tensor = torch.tensor([0.0, 1.0]),
    return_intermediates: bool = False,
    verbose: bool = False,
    enable_grad: bool = False,
    **model_extras,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Solve the Riemannian ODE and, in addition, compute the integrated divergence.

    This method is similar to `sample`, but during the integration it computes the divergence
    (i.e. the trace of the Jacobian of the projected velocity field) at each time step and
    accumulates dt * divergence to approximate the log determinant of the transformation.

    Args:
        x_init (Tensor): initial conditions.
        step_size (float): step size.
        return_div (bool): If True, return a tuple (final_sample, log_det); otherwise, return sample only.
        projx (bool): Whether to project the state onto the manifold at each step.
        proju (bool): Whether to project the velocity onto the tangent plane at each step.
        method (str): One of ["euler", "midpoint", "rk4"].
        time_grid (Tensor): Time grid (will be sorted). If step_size is not None, a uniform discretization is used.
        return_intermediates (bool): If True, return samples at the times specified in time_grid.
        verbose (bool): Whether to display a progress bar.
        enable_grad (bool): Whether to compute gradients during sampling.
        **model_extras: Additional keyword arguments for the velocity model.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: If return_div is True, returns a tuple containing the final sample (or sequence if return_intermediates is True)
        and the final log determinant; otherwise, returns the final sample only.
    """
    # Select the integration step function.
    step_fns = {
        "euler": _euler_step,
        "midpoint": _midpoint_step,
        "rk4": _rk4_step,
    }

    if solver.manifold not in [BallManifold, ConvexPolytopeManifold]

    assert method in step_fns.keys(), f"Unknown method {method}"
    step_fn = step_fns[method]

    # Define a velocity function that includes any necessary projections.
    def velocity_func(x, t):
        return self.velocity_model(x=x, t=t, **model_extras)

    # Prepare the time discretization.
    time_grid = torch.sort(time_grid.to(device=x_init.device)).values
    if step_size is None:
        t_discretization = time_grid
    else:
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        assert (t_final - t_init) > step_size, (
            f"Time interval [{t_init}, {t_final}] must be larger than step_size {step_size}."
        )
        n_steps = math.ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )
    t0s = t_discretization[:-1]
    if verbose:
        t0s = tqdm(t0s)

    # Initialize divergence accumulator.
    log_det = torch.zeros(x_init.shape[0], device=x_init.device)

    # Optionally collect intermediate states.
    if return_intermediates:
        xts = []
        i_ret = 0
    with torch.set_grad_enabled(True):
        xt = x_init
        for t0, t1 in zip(t0s, t_discretization[1:]):
            dt = t1 - t0

            # --- Compute divergence of the velocity field at the current state.
            # We need gradients here even if enable_grad is False,
            # so we use a context that always allows grad.

            xt.requires_grad_(True)
            vt = velocity_func(xt, t0)
            # Compute the exact divergence as the trace of the Jacobian of vt w.r.t. xt.
            divergence = 0.0
            # Assuming xt has shape [batch, d]
            d = xt.shape[1]
            for i in range(d):
                grad_i = torch.autograd.grad(
                    vt[:, i].sum(), xt, retain_graph=True, create_graph=True
                )[0][:, i]
                divergence = divergence + grad_i
            # Accumulate the divergence contribution.
            log_det = log_det + dt * divergence

            # --- Take the integration step.
            xt_next = step_fn(
                velocity_func,
                xt,
                t0,
                dt,
                manifold=self.manifold,
                projx=projx,
                proju=proju,
            )

            # --- If returning intermediates, interpolate between xt and xt_next.
            if return_intermediates:
                while (
                    i_ret < len(time_grid)
                    and t0 <= time_grid[i_ret]
                    and time_grid[i_ret] <= t1
                ):
                    xts.append(
                        interp(self.manifold, xt, xt_next, t0, t1, time_grid[i_ret])
                    )
                    i_ret += 1
            xt = xt_next.detach()  # detach to prevent unnecessary graph buildup

    if return_intermediates:
        final_sample = torch.stack(xts, dim=0)
    else:
        final_sample = xt

    if return_div:
        return final_sample, log_det
    else:
        return final_sample


def riem_compute_likelihood(
        riem_solver: RiemannianODESolver,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: float,
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
    """Compute log-likelihood by solving the ODE backwards on the manifold."""
    assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
    ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

    # For stochastic divergence estimation
    if not exact_divergence:
        z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0
        z = riem_solver.manifold.proju(x_1, z) if proju else z

    def dynamics_func(x, t):
        with torch.set_grad_enabled(True):
            x.requires_grad_()
            u = riem_solver.velocity_model(x=x, t=t, **model_extras)
            if proju:
                u = riem_solver.manifold.proju(x, u)

            if exact_divergence:
                div = 0
                for i in range(u.flatten(1).shape[1]):
                    ei = torch.zeros_like(u)
                    ei[:, i] = 1
                    if proju:
                        ei = riem_solver.manifold.proju(x, ei)

                    div_i = gradient(u[:, i], x, create_graph=True)[:, i]
                    christoffel = riem_solver.manifold.christoffel(x, ei, u) if hasattr(riem_solver.manifold, 'christoffel') else 0
                    div += div_i - christoffel
            else:
                u_dot_z = riem_solver.manifold.inner(x, u, z)
                grad_u_dot_z = gradient(u_dot_z, x)
                if proju:
                    grad_u_dot_z = riem_solver.manifold.proju(x, grad_u_dot_z)
                div = riem_solver.manifold.inner(x, grad_u_dot_z, z)

        return u.detach(), div.detach()

    step_fns = {
        "euler": _euler_step,
        "midpoint": _midpoint_step,
        "rk4": _rk4_step,
    }
    assert method in step_fns.keys(), f"Unknown method {method}"
    step_fn = step_fns[method]

    # Prepare time discretization for backward integration
    t_init, t_final = time_grid[0].item(), time_grid[-1].item()  # 1.0, 0.0
    if step_size is None:
        t_discretization = time_grid
    else:
        n_steps = math.ceil((t_init - t_final) / step_size)  # Note: t_init > t_final
        t_discretization = torch.linspace(t_init, t_final, n_steps + 1, device=x_1.device)

    if return_intermediates:
        path = []
        log_det = []
        i_ret = 0

    with torch.set_grad_enabled(enable_grad):
        xt = x_1
        accum_div = torch.zeros(x_1.shape[0], device=x_1.device)

        t0s = t_discretization[:-1]
        if verbose:
            t0s = tqdm(t0s)

        for t0, t1 in zip(t0s, t_discretization[1:]):
            dt = t1 - t0  # Will be negative for backward integration

            ut, div = dynamics_func(xt, t0)
            accum_div = accum_div + div * (-dt)  # Note the negative dt for backward integration

            xt_next = step_fn(
                lambda x, t: -ut,
                xt,
                t0,
                -dt,  # Make dt positive for the step function
                manifold=riem_solver.manifold,
                projx=projx,
                proju=proju,
            )

            if return_intermediates:
                while (
                        i_ret < len(time_grid)
                        and t0 >= time_grid[i_ret]  # Changed condition for backward integration
                        and time_grid[i_ret] >= t1
                ):
                    interp_x = interp(
                        riem_solver.manifold, xt, xt_next, t0, t1, time_grid[i_ret]
                    )
                    path.append(interp_x)
                    log_det.append(accum_div)
                    i_ret += 1

            xt = xt_next

    source_log_p = log_p0(xt)
    final_log_p = source_log_p + accum_div

    if return_intermediates:
        return torch.stack(path, dim=0), torch.stack(log_det, dim=0)
    else:
        return xt, final_log_p[:, None]


if __name__ == "__main__":
    device='cpu'
    import numpy as np
    from manifolds import BallManifold
    from scipy.special import gammaln

    torch_linalg = LinAlg(backend='torch', device=device, dtype=np.float32)
    def sample_ball(shape):
        return torch_linalg.sample_unit_hyper_sphere_ball(shape, ball=True)

    K = 4
    log_ball_vol = (K / 2) * np.log(np.pi) - gammaln(K / 2 + 1)
    def log_p_ball(values):
        return torch.full(values.shape[:-1], -log_ball_vol, dtype=values.dtype, device=values.device)

    # aff_model = MLP(input_dim=4, time_dim=1, hidden_dim=512).to(device)
    # aff_model.load_state_dict(torch.load(r"C:\python_projects\sbmfi\arxiv_polytope\aff_model.pt"))
    # wrapped_vf = WrappedModel(aff_model)

    # solver = ODESolver(velocity_model=wrapped_vf)
    # time_grid = torch.Tensor([0.0, 1.0]).to(dtype=torch.float, device=device)
    # x_init = sample_ball((50,4))
    # dang = solver.sample(x_init, step_size=0.05, method='midpoint')
    # ding = sample_and_div(
    #         ode_solver=solver,
    #         x_init=x_init,
    #         time_grid=time_grid,
    #         step_size=0.05,
    #         method= "midpoint",
    #         return_intermediates = False,
    #         return_div= True,
    #         exact_divergence=True,
    # )

    manifold = BallManifold(dim=4)
    ball_model = ProjectToTangent(  # Ensures we can just use Euclidean divergence.
        MLP(  # Vector field in the ambient space.
            input_dim=4,
            hidden_dim=512,
        ),
        manifold=manifold,
    ).to(device)

    ball_model.load_state_dict(torch.load(r"C:\python_projects\sbmfi\arxiv_polytope\ball_model.pt"))

    step_size = torch.Tensor([0.05]).to(dtype=torch.float, device=device)

    x_init = sample_ball((40, 4))
    wrapped_vf = WrappedModel(ball_model)

    solver = RiemannianODESolver(velocity_model=wrapped_vf, manifold=manifold)  # create an ODESolver class

    # sol = solver.sample(x_init, 0.05)
    sol2, logq = riem_sample_and_div(solver, x_init, step_size=0.05, return_log_p=True, exact_divergence=True)


