# this is for the arxiv_polytope publication
import torch
import torch
from torch.distributions import MultivariateNormal, Categorical

class MixtureOfGaussians(torch.distributions.Distribution):
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
        """
        Initialize a mixture of Gaussians distribution.

        :param means: Tensor of shape (num_components, num_dimensions) representing the means of the components.
        :param covariances: Tensor of shape (num_components, num_dimensions, num_dimensions) representing the covariance matrices.
        :param weights: Tensor of shape (num_components,) representing the mixture weights. Should sum to 1.
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights

        # Number of components and dimensions
        self.num_components = means.shape[0]
        self.num_dimensions = means.shape[1]

        # Ensure weights sum to 1
        if not torch.isclose(weights.sum(), torch.tensor(1.0)):
            raise ValueError("Mixture weights must sum to 1.")

        # Define the categorical distribution for component selection
        self.categorical = Categorical(weights)

        # Define multivariate normal distributions for each component
        self.components = [
            MultivariateNormal(means[i], covariances[i])
            for i in range(self.num_components)
        ]

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the mixture distribution.

        :param sample_shape: Shape of the samples to generate.
        :return: Tensor of samples with shape (*sample_shape, num_dimensions).
        """
        num_samples = sample_shape.numel() if sample_shape else 1

        # Sample component indices
        component_indices = self.categorical.sample(sample_shape)

        # Sample from the corresponding Gaussian components
        samples = torch.empty(num_samples, self.num_dimensions)
        for i in range(self.num_components):
            mask = component_indices == i
            num_samples_i = mask.sum().item()
            if num_samples_i > 0:
                samples[mask] = self.components[i].sample((num_samples_i,))

        return samples.squeeze(0) if sample_shape == torch.Size() else samples

    def log_prob(self, value):
        """
        Compute the log probability of a value under the mixture distribution.

        :param value: Tensor of shape (..., num_dimensions).
        :return: Tensor of log probabilities with shape (...,).
        """
        # Compute log probabilities for each component
        log_probs = torch.stack(
            [comp.log_prob(value) + torch.log(self.weights[i]) for i, comp in enumerate(self.components)],
            dim=-1
        )
        # Marginalize over components
        return torch.logsumexp(log_probs, dim=-1)

if __name__ == "__main__":
    # Example usage
    means = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    covariances = torch.stack([torch.eye(2), torch.eye(2) * 2])
    weights = torch.tensor([0.4, 0.6])

    mog = MixtureOfGaussians(means, covariances, weights)

    # Sample from the mixture
    samples = mog.sample((1000,))

    # Compute log probability of a value
    log_prob = mog.log_prob(torch.tensor([1.0, 1.0]))
    print("Log Probability:", log_prob)
