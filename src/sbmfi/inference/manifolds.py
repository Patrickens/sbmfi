import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flow_matching.utils.manifolds import Manifold

###############################################################################
#  Maximum-Entropy Polytope Manifold
###############################################################################

class MaxEntPolytopeManifold(Manifold):
    """
    A manifold representing a convex polytope via maximum entropy coordinates.

    The polytope is defined as the convex hull of a set of vertices \(V \in \mathbb{R}^{m \times n}\):
        \[
            \mathcal{P} = \Big\{ x \in \mathbb{R}^n \;:\; x = \sum_{i=1}^m \lambda_i\, V_i,\; \lambda \in \Delta^{m-1} \Big\}.
        \]

    For a given point \(x\) (assumed to lie in the polytope), we define an approximate maximum
    entropy coordinate mapping by
        \[
           \lambda_i(x) \approx \frac{\exp\Big(-\|x-V_i\|^2/\sigma\Big)}
           {\sum_{j=1}^m \exp\Big(-\|x-V_j\|^2/\sigma\Big)}.
        \]

    The inverse mapping is given by
        \[
           x = \sum_{i=1}^m \lambda_i(x) \, V_i.
        \]

    In the space of maximum-entropy coordinates (which is the interior of the probability simplex),
    a natural (Fisher–Rao) exponential map is given by mapping to the “natural parameters” via
        \(\log\lambda\), adding a tangent update, and then mapping back via a softmax.

    **Notes:**
    - For expmap we assume that the provided tangent vector \(u\) (of shape \((B, m)\)) lies in the
      tangent space of the simplex (i.e. it satisfies \(\sum_i u_i=0\) for each example).
    - For simplicity, the `projx` method simply returns \(x\) (assuming that \(x\) is already in the polytope).
    """

    def __init__(self, vertices: Tensor, sigma: float = 1.0):
        """
        Args:
            vertices (Tensor): A tensor of shape \((m, n)\) containing the vertices of the polytope.
            sigma (float): Temperature parameter controlling the “sharpness” of the softmax.
        """
        super(MaxEntPolytopeManifold, self).__init__()
        # Save the vertices; these define the polytope as conv(V).
        self.register_buffer('V', vertices)  # shape (m, n)
        self.sigma = sigma

    def maxent_coords(self, x: Tensor) -> Tensor:
        """
        Compute approximate maximum entropy coordinates for a batch of points.

        Args:
            x (Tensor): Tensor of shape \((B, n)\) of points in the polytope.

        Returns:
            Tensor: Tensor of shape \((B, m)\) with \(\lambda_i(x)\) approximated by
            \(\text{softmax}_i\Big(-\|x-V_i\|^2/\sigma\Big)\).
        """
        B, n = x.shape
        m = self.V.shape[0]
        # Expand dimensions to compute pairwise squared distances:
        #   x_exp: (B, 1, n), V_exp: (1, m, n)
        x_exp = x.unsqueeze(1)  # shape: (B, 1, n)
        V_exp = self.V.unsqueeze(0)  # shape: (1, m, n)
        diff = x_exp - V_exp  # shape: (B, m, n)
        d2 = (diff ** 2).sum(dim=-1)  # shape: (B, m)
        # Compute softmax of the negative distances (divided by sigma).
        lambda_coords = F.softmax(-d2 / self.sigma, dim=-1)  # shape: (B, m)
        return lambda_coords

    def inv_maxent(self, lambda_coords: Tensor) -> Tensor:
        """
        Map from maximum entropy coordinates back to the polytope.

        Args:
            lambda_coords (Tensor): Tensor of shape \((B, m)\) with coordinates.
        Returns:
            Tensor: Tensor of shape \((B, n)\) computed as \(x = \lambda\,V\).
        """
        return lambda_coords.matmul(self.V)  # (B, n)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Exponential map on the manifold.

        Interpreted in maximum entropy coordinates: given a base point \(x\) with
        coordinates \(\lambda(x)\) and a tangent vector \(u\) in the natural parameter space,
        we define
            \[
            \lambda_{\text{new}} = \operatorname{softmax}\big( \log \lambda(x) + u\big),
            \]
        and then map back by
            \[
            x_{\text{new}} = \sum_i [\lambda_{\text{new}}]_i\, V_i.
            \]

        Args:
            x (Tensor): Tensor of shape \((B, n)\), base point(s) in the polytope.
            u (Tensor): Tensor of shape \((B, m)\) in the tangent space (satisfying \(\sum_i u_i = 0\)).
        Returns:
            Tensor: Tensor of shape \((B, n)\), the resulting point on the manifold.
        """
        # Convert x to maximum entropy coordinates.
        lambda_x = self.maxent_coords(x)  # shape: (B, m)
        # Map to natural parameter space.
        log_lambda = torch.log(lambda_x + 1e-12)
        # Add the tangent vector.
        log_lambda_new = log_lambda + u
        # Map back to the simplex.
        lambda_new = F.softmax(log_lambda_new, dim=-1)
        # Recover the new point.
        x_new = self.inv_maxent(lambda_new)
        return self.projx(x_new)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map on the manifold.

        Computes a tangent vector \(u\) (in the natural parameter space) at \(x\) that “points toward” \(y\).
        In our approximation, we define
            \[
            u = \log \lambda(y) - \log \lambda(x),
            \]
        and then subtract the mean so that \(u\) lies in the tangent space of the simplex.

        Args:
            x (Tensor): Tensor of shape \((B, n)\), base point.
            y (Tensor): Tensor of shape \((B, n)\), target point.
        Returns:
            Tensor: Tensor of shape \((B, m)\), the tangent vector in the natural parameter space.
        """
        lambda_x = self.maxent_coords(x)
        lambda_y = self.maxent_coords(y)
        log_lambda_x = torch.log(lambda_x + 1e-12)
        log_lambda_y = torch.log(lambda_y + 1e-12)
        u = log_lambda_y - log_lambda_x
        # Ensure the result lies in the tangent space (i.e. sum(u) = 0).
        u = u - u.mean(dim=-1, keepdim=True)
        return u

    def projx(self, x: Tensor) -> Tensor:
        """
        Project a point \(x\) onto the polytope.

        Here we assume that \(x\) is already (approximately) in the convex hull of \(V\).
        A more robust implementation might solve a quadratic program to find the closest point
        in conv(\(V\)) to \(x\), but for simplicity we return \(x\) unchanged.

        Args:
            x (Tensor): Tensor of shape \((B, n)\).
        Returns:
            Tensor: Tensor of shape \((B, n)\).
        """
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Project a tangent vector \(u\) onto the tangent space at \(x\).

        In the space of maximum entropy coordinates (i.e. the interior of the probability simplex),
        the tangent space is the hyperplane \(\{u \in \mathbb{R}^m : \sum_i u_i = 0\}\). We ensure this by
        subtracting the mean.

        Args:
            x (Tensor): Tensor of shape \((B, n)\) (unused here).
            u (Tensor): Tensor of shape \((B, m)\), the candidate tangent vector.
        Returns:
            Tensor: Tensor of shape \((B, m)\), projected onto the tangent space.
        """
        return u - u.mean(dim=-1, keepdim=True)


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


import abc
import math
import torch
import torch.nn as nn
from torch import Tensor



########################################################
#  Poincaré Ball Manifold
########################################################
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
        # Radius = 1 / sqrt(c)
        maxnorm = 1.0 / math.sqrt(self.c)
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm_x > maxnorm
        # Scale points that are outside the ball
        safe_x = x.clone()
        safe_x[cond] = (
                               safe_x[cond] / norm_x[cond]
                       ) * (maxnorm - self.eps)
        return safe_x

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


if __name__ == "__main__":
    pass