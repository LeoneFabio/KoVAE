import numpy as np
import torch
from viz.latent_traversals import LatentTraverser
from scipy import stats
from utils.utils_model import get_device


class Visualizer():
    def __init__(self, model):
        
        self.model = model
        self.latent_traverser = LatentTraverser(self.model.latent_spec)

    def reconstructions(self, data, time, final_index):
        # Compute reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        recon_data = self.model(data['inter'], time, final_index, isTraining=False)[0]
        self.model.train()
        return recon_data
    
    def get_original_data(self, data):
        x = data['data'].to(get_device()).float()
        x = x[:, :, :-1]
        return x
        

    def samples(self, size=(8, 8), filename='samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8):
        """
        Generates a record traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        return generated

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def all_latent_traversals(self, size=8):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = Variable(latent_samples)
        if self.model.use_cuda:
            latent_samples = latent_samples.cuda()
        return self.model.decode(latent_samples).cpu()