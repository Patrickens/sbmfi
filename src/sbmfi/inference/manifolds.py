import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flow_matching.utils.manifolds import Manifold
import math


class MaxEntManifold(Manifold):
    def __init__(self, vertices: torch.Tensor, tol: float = 1e-8, max_iter: int = 100):
        """
        Args:
            vertices (Tensor): a tensor of shape (n, d) containing the vertices (e.g. for a hypercube).
            tol (float): tolerance for the root solver.
            max_iter (int): maximum number of iterations for LBFGS.
        """
        super().__init__()
        # Save vertices as a buffer so that they are moved to the proper device automatically.
        self.register_buffer("vertices", vertices)
        self.tol = tol
        self.max_iter = max_iter

    def _solve_mu(self, x: torch.Tensor):
        r"""Given a point (or batch of points) x, solve for the dual variable μ such that

            f(μ) = \sum_i \lambda_i v_i - x = 0,

        where
            \lambda_i = exp(v_i \cdot μ) / \sum_j exp(v_j \cdot μ).

        This is done by minimizing the objective
            L(μ) = 0.5 * || (\sum_i \lambda_i v_i - x) ||^2.

        Args:
            x (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            mu (Tensor): the computed dual variable(s), shape (B,d)
            lambdas (Tensor): the maximum entropy coordinates, shape (B,n)
        """
        B, d = x.shape
        # Initialize μ with zeros; one per batch element.
        mu = torch.zeros(B, d, device=x.device, dtype=x.dtype, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [mu],
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            # Compute dot products: (B, n)
            dot = torch.matmul(mu, self.vertices.T)
            exp_dot = torch.exp(dot)
            # Compute partition function Z: (B,1)
            Z = torch.sum(exp_dot, dim=1, keepdim=True)
            # Compute lambdas: (B, n)
            lambdas = exp_dot / Z
            # Reconstruct x from lambdas: (B, d)
            rec = torch.matmul(lambdas, self.vertices)
            loss = 0.5 * torch.sum((rec - x) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)

        # with torch.no_grad():
        dot = torch.matmul(mu, self.vertices.T)
        exp_dot = torch.exp(dot)
        Z = torch.sum(exp_dot, dim=1, keepdim=True)
        lambdas = exp_dot / Z

        # return mu.detach(), lambdas.detach()
        return mu, lambdas

    def max_entropy_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the maximum entropy (barycentric) coordinates for point(s) x.

        Args:
            x (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: maximum entropy coordinates (lambdas), shape (B, n)
        """
        _, lambdas = self._solve_mu(x)
        return lambdas

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct the point from its maximum entropy coordinates.

        Args:
            x (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: reconstructed point(s), shape (B,d)
        """
        lambdas = self.max_entropy_coordinates(x)
        return torch.matmul(lambdas, self.vertices)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Computes the exponential map on the MaxEntManifold.

        The idea is to compute the dual coordinate μ corresponding to x,
        add the tangent vector u, and then map back to the manifold.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            u (Tensor): tangent vector(s) at x, shape (B,d)

        Returns:
            Tensor: the point(s) expₓ(u) on the manifold, shape (B,d)
        """
        mu_x, _ = self._solve_mu(x)  # Get the dual coordinate for x.
        mu_new = mu_x + u  # Move in the dual space.
        dot = torch.matmul(mu_new, self.vertices.T)
        exp_dot = torch.exp(dot)
        Z = torch.sum(exp_dot, dim=1, keepdim=True)
        lambdas = exp_dot / Z
        x_new = torch.matmul(lambdas, self.vertices)
        return x_new

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computes the logarithmic map on the MaxEntManifold.

        We define the log map as the difference of the dual coordinates:
            logₓ(y) = μ_y - μₓ.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            y (Tensor): target point(s) on the manifold, shape (B,d)

        Returns:
            Tensor: tangent vector(s) at x, shape (B,d)
        """
        mu_x, _ = self._solve_mu(x)
        mu_y, _ = self._solve_mu(y)
        return mu_y - mu_x

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        r"""Projects an arbitrary point x to the manifold.

        In this simple implementation we assume that x is already in the interior
        of the convex hull defined by the vertices.

        Args:
            x (Tensor): point(s) to be projected, shape (B,d)

        Returns:
            Tensor: projected point(s), shape (B,d)
        """
        # For a general polytope one might solve a quadratic program.
        # Here we assume that x is already valid.
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Projects an arbitrary tangent vector u at x onto the tangent space.

        For points in the interior of the convex hull, the tangent space is all of ℝᵈ.

        Args:
            x (Tensor): base point(s) on the manifold, shape (B,d)
            u (Tensor): vector(s) to be projected, shape (B,d)

        Returns:
            Tensor: projected tangent vector(s), shape (B,d)
        """
        # In the interior, the tangent space is all of ℝᵈ.
        return u


class BallManifold(Manifold):
    """
    A Euclidean ball manifold representing the open ball in ℝ^n:
        { x in ℝ^n : ||x|| < radius }.

    The exponential map is defined as:
        expmap(x, u) = projx(x + u),
    and the logarithmic map as:
        logmap(x, y) = y - x.

    The projection operation ensures that any point that would fall outside
    the open ball is scaled back to lie inside.
    """

    def __init__(self, dim: int, radius: float = 1.0, eps: float = 1e-5):
        """
        Args:
            dim (int): Dimension of the ambient space ℝ^n.
            radius (float): Radius of the ball.
            eps (float): A small constant to ensure points remain strictly inside the ball.
        """
        super(BallManifold, self).__init__()
        self.dim = dim
        self.radius = radius
        self.eps = eps

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        For the Euclidean open ball, the exponential map is given by addition,
        followed by a projection to enforce the ball constraint.
        """
        y = x + u
        return self.projx(y)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        The logarithmic map in the Euclidean setting is simply subtraction.
        """
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        """
        Projects points onto the open ball by scaling any point that lies outside
        (or too close to the boundary) back to have norm (radius - eps).

        Args:
            x (Tensor): Tensor of shape (..., n) representing points in ℝ^n.

        Returns:
            Tensor: The projected points, each with norm at most (radius - eps).
        """
        # Compute the Euclidean norm along the last dimension.
        norm = x.norm(p=2, dim=-1, keepdim=True)
        # For points with norm greater than (radius - eps), compute the scaling factor.
        factor = torch.where(norm > (self.radius - self.eps),
                             (self.radius - self.eps) / (norm + 1e-12),
                             torch.ones_like(norm))
        return factor * x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        In Euclidean space, the tangent space at any point is ℝ^n, so the tangent
        vector u is already valid.
        """
        return u


class PoincareBallManifold(nn.Module):
    """
    Implements operations on the n-dimensional Poincaré ball model
    of hyperbolic space with negative curvature -c.

    Attributes:
        c (float): positive curvature parameter (default: 1.0).
        eps (float): small epsilon for numerical stability.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        # Curvature c > 0 => negative curvature -c in the Poincaré model
        self.c = c
        self.eps = eps

    ########################################################
    #  Utility Functions
    ########################################################
    def _norm(self, x: Tensor, keepdim: bool = True) -> Tensor:
        """Compute Euclidean norm of x along last dimension."""
        return torch.norm(x, p=2, dim=-1, keepdim=keepdim).clamp_min(self.eps)

    def _lambda_x(self, x: Tensor) -> Tensor:
        r"""
        Compute the conformal factor:
            λ_x^c = 2 / (1 - c * ||x||^2).
        Used in exponential/logarithmic maps.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp_min(self.eps)
        return 2.0 / (1.0 - self.c * x2).clamp_min(self.eps)

    def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Möbius addition in the Poincaré ball (with curvature c):

        x ⊕_c y =
          ( (1 + 2c <x,y> + c||y||^2) x + (1 - c||x||^2) y )
          ------------------------------------------------
                       (1 + 2c <x,y> + c^2 ||x||^2 ||y||^2)
        """
        xy = torch.sum(x * y, dim=-1, keepdim=True)  # <x,y>
        x2 = torch.sum(x * x, dim=-1, keepdim=True)  # ||x||^2
        y2 = torch.sum(y * y, dim=-1, keepdim=True)  # ||y||^2

        num = (1.0 + 2.0 * self.c * xy + self.c * y2) * x + (1.0 - self.c * x2) * y
        denom = 1.0 + 2.0 * self.c * xy + (self.c ** 2) * x2 * y2
        return num / denom.clamp_min(self.eps)

    def _mobius_neg(self, x: Tensor) -> Tensor:
        """Möbius 'negative' just the usual -x in R^n, used in logmap."""
        return -x

    ########################################################
    #  Required Methods
    ########################################################
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        r"""
        Exponential map at point x applied to tangent vector u.

        Formula (c > 0):
            exp_x^c(u) = x ⊕_c(
                tanh( sqrt(c)/2 * λ_x^c * ||u|| ) * (u / ( sqrt(c)*||u|| ))
            )

        where λ_x^c = 2 / (1 - c ||x||^2).

        Args:
            x (Tensor): BxN manifold points (on Poincaré ball).
            u (Tensor): BxN tangent vectors at x.

        Returns:
            Tensor: BxN points in the Poincaré ball.
        """
        norm_u = self._norm(u)  # ||u||
        lambda_x = self._lambda_x(x)  # λ_x^c

        # scale = tanh( sqrt(c)/2 * λ_x^c * ||u|| )
        scaled_factor = torch.tanh(
            torch.sqrt(torch.tensor(self.c, device=x.device)) * lambda_x * norm_u / 2.0
        )

        # direction = u / (||u|| * sqrt(c))
        direction = u / norm_u / torch.sqrt(torch.tensor(self.c, device=x.device))

        gamma = scaled_factor * direction  # same shape as x, y
        return self._mobius_add(x, gamma)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Logarithmic map at x of point y on the ball.

        Formula (c > 0):
            log_x^c(y) =
                2 / ( sqrt(c) * λ_x^c ) * artanh( sqrt(c) * ||-x ⊕_c y|| )
                * ( (-x ⊕_c y) / ||-x ⊕_c y|| )

        Args:
            x (Tensor): BxN manifold points (on Poincaré ball).
            y (Tensor): BxN manifold points (on Poincaré ball).

        Returns:
            Tensor: BxN tangent vectors at x.
        """
        # Möbius subtract is mobius_add(x, -y) or the other way around.
        # But for log, we compute z = (-x) ⊕_c y
        z = self._mobius_add(self._mobius_neg(x), y)
        norm_z = self._norm(z)

        lambda_x = self._lambda_x(x)  # λ_x^c

        # factor = 2/( sqrt(c)*λ_x^c ) * artanh( sqrt(c)*||z|| )
        # direction = z / ||z||
        scale = (
                        2.0
                        / (
                                torch.sqrt(torch.tensor(self.c, device=x.device)) * lambda_x
                        ).clamp_min(self.eps)
                ) * torch.atanh(
            torch.sqrt(torch.tensor(self.c, device=x.device)) * norm_z.clamp_max(1 - self.eps)
        ).clamp_min(self.eps)

        direction = z / norm_z
        return scale * direction

    def projx(self, x: Tensor) -> Tensor:
        """
        Project points x onto the open Poincaré ball of radius 1/sqrt(c).
        By default, we ensure the norm < 1/sqrt(c).
        """
        maxnorm = 1.0 / math.sqrt(self.c)
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        # cond has shape [B, 1]. We want [B] for row-wise indexing in x.
        cond = (norm_x > maxnorm).squeeze(-1)  # shape [B]

        safe_x = x.clone()
        # safe_x[cond] has shape [num_cond_true, N]
        # norm_x[cond] has shape [num_cond_true, 1]
        # which will broadcast to [num_cond_true, N]
        safe_x[cond] = (
                               safe_x[cond] / norm_x[cond]
                       ) * (maxnorm - self.eps)
        return safe_x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Project a tangent vector u onto the tangent space at x.
        For the Poincaré disk, the tangent space at x is isomorphic to R^n,
        so we usually do not need a special projection (identity).
        """
        # Typically identity for Poincaré ball, but
        # you could also multiply by the conformal factor if needed for
        # certain gradient adjustments. For now, do identity.
        return u


class ConvexPolytopeManifold(nn.Module):
    r"""
    A manifold defined by an arbitrary convex polytope

        P = { x in R^n : A x <= b }.

    Since P may not be smooth at the boundary, we define the following operations
    using the Euclidean metric with projections:

      - Exponential map:  exp_x(u) = projx(x + u)
      - Logarithmic map:  log_x(y) = y - x

    Both the point projection and tangent-vector projection are implemented via solving
    a quadratic program in dual form using batched projected gradient descent.
    """

    def __init__(
            self,
            A: Tensor,
            b: Tensor,
            proj_iters: int = 50,
            proju_iters: int = 10,
            tol: float = 1e-5,
            proj_step: float = 1e-2,
            proju_step: float = 1e-2,
    ):
        """
        Args:
            A (Tensor): Constraint matrix of shape [m, n].
            b (Tensor): Constraint right-hand side, shape [m].
            proj_iters (int): Number of iterations for the point projection.
            proju_iters (int): Number of iterations for the tangent projection.
            tol (float): Tolerance for determining constraint activeness.
            proj_step (float): Step size for dual updates in projx.
            proju_step (float): Step size for dual updates in proju.
        """
        super().__init__()
        # Store constraints as buffers
        self.register_buffer("A", A)  # shape [m, n]
        self.register_buffer("b", b)  # shape [m]
        self.proj_iters = proj_iters
        self.proju_iters = proju_iters
        self.tol = tol
        self.proj_step = proj_step
        self.proju_step = proju_step
        # Precompute Q = A A^T (which appears in both dual problems)
        Q = A @ A.t()  # shape [m, m]
        self.register_buffer("Q", Q)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        r"""Exponential map defined by
            exp_x(u) = projx(x + u)
        """
        return self.projx(x + u)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Logarithmic map (using Euclidean difference):
            log_x(y) = y - x.
        """
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        r"""
        Projects a batch of points x (shape [B, n]) onto the convex polytope

            P = { x in R^n : A x <= b }.

        This is achieved by solving the dual quadratic program

            min_{λ ≥ 0}  ½ λᵀ Q λ - λᵀ (A x - b)
        with Q = A Aᵀ. Once λ is computed (approximately), the projection is given by

            projx(x) = x - Aᵀ λ.
        """
        B, n = x.shape
        m = self.A.shape[0]
        # Compute c = A x - b for each sample.
        # x: [B, n], A: [m, n]  -->  x @ A.T: [B, m]
        c = x @ self.A.t() - self.b.unsqueeze(0)  # shape: [B, m]
        # Initialize dual variables λ as zeros.
        lam = torch.zeros(B, m, device=x.device, dtype=x.dtype)
        # Batched projected gradient descent on the dual problem.
        for _ in range(self.proj_iters):
            # Gradient: grad = λ @ Q - c.
            grad = lam @ self.Q - c  # shape: [B, m]
            lam = torch.clamp(lam - self.proj_step * grad, min=0.0)
        # Recover the projected point: z = x - Aᵀ λ.
        z = x - lam @ self.A  # shape: [B, n]
        return z

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        r"""
        Projects a tangent vector u at x onto the tangent cone of P at x.

        For points on the boundary, the feasible directions v must satisfy
            A_i v ≤ 0   for every active constraint (i.e. when A_i x ≥ b_i - tol).

        We solve the following dual problem for each sample:

            min_{λ ≥ 0}  ½ λᵀ Q λ - λᵀ ( (u @ Aᵀ) masked by activeness )

        and then set:

            proju(x, u) = u - Aᵀ λ.

        The “active” constraints are determined by checking which rows of A satisfy
            A x ≥ b - tol.
        """
        B, n = x.shape
        m = self.A.shape[0]
        # Compute A x for each sample.
        Ax = x @ self.A.t()  # shape: [B, m]
        # Determine activeness: active if A_i x >= b_i - tol.
        active = (Ax >= (self.b.unsqueeze(0) - self.tol)).to(u.dtype)  # shape: [B, m]
        # For each sample, form the masked right-hand side: (u @ A.T) elementwise multiplied by active.
        masked = (u @ self.A.t()) * active  # shape: [B, m]
        # Initialize dual variables for the tangent projection.
        lam = torch.zeros(B, m, device=x.device, dtype=u.dtype)
        for _ in range(self.proju_iters):
            grad = lam @ self.Q - masked  # shape: [B, m]
            lam = torch.clamp(lam - self.proju_step * grad, min=0.0)
            # Ensure that λ remains zero for constraints that are inactive.
            lam = lam * active
        # Compute the projected tangent vector.
        u_proj = u - lam @ self.A  # shape: [B, n]
        return u_proj


if __name__ == "__main__":
    pass