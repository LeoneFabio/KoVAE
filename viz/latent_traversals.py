import numpy as np
import torch
from scipy import stats


class LatentTraverser:
    def __init__(self, latent_spec):
        """
        LatentTraverser is used to generate traversals of the latent space.

        Parameters
        ----------
        latent_spec : dict
            Dictionary with keys 'cont' (int) and optionally 'disc' (list of ints).
            Example: {'cont': 10, 'disc': [3, 4]}.
        """
        self.latent_spec = latent_spec
        self.sample_prior = False

        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.cont_dim = latent_spec['cont'] if self.is_continuous else 0
        self.disc_dims = latent_spec['disc'] if self.is_discrete else []

    def traverse_line(self, cont_idx=None, disc_idx=None, size=8):
        """
        Returns a (size, latent_dim) tensor by traversing one continuous or discrete latent.

        Parameters
        ----------
        cont_idx : int or None
        disc_idx : int or None
        size : int
        """
        samples = []

        if self.is_continuous:
            samples.append(self._traverse_continuous_line(cont_idx, size))

        if self.is_discrete:
            for i, dim in enumerate(self.disc_dims):
                traverse = (i == disc_idx)
                samples.append(self._traverse_discrete_line(dim, traverse, size))

        return torch.cat(samples, dim=1)

    def _traverse_continuous_line(self, idx, size):
        """
        Returns (size, cont_dim) tensor with traversal on dimension `idx`.
        """
        samples = np.random.normal(size=(size, self.cont_dim)) if self.sample_prior else np.zeros((size, self.cont_dim))

        if idx is not None:
            cdf_vals = np.linspace(0.05, 0.95, size)
            traversal_values = stats.norm.ppf(cdf_vals)
            for i in range(size):
                samples[i, idx] = traversal_values[i]

        return torch.tensor(samples, dtype=torch.float32)

    def _traverse_discrete_line(self, dim, traverse, size):
        """
        Returns (size, dim) one-hot tensor for discrete variable.
        """
        samples = np.zeros((size, dim))

        if traverse:
            for i in range(size):
                samples[i, i % dim] = 1.0
        else:
            if self.sample_prior:
                random_indices = np.random.randint(0, dim, size)
                samples[np.arange(size), random_indices] = 1.0
            else:
                samples[:, 0] = 1.0  # deterministic default

        return torch.tensor(samples, dtype=torch.float32)
    
    
