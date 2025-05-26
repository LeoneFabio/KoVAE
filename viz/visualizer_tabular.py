import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from viz.latent_traversals import LatentTraverser
from viz.getter import Getter

class TabularVisualizer:
    def __init__(self, model, output_dir='./visualizations', save_plots=True):
        """
        Visualizer for KoVAE applied to tabular time series data.
        """
        self.model = model
        self.latent_traverser = LatentTraverser(model.latent_spec)
        self.getter = Getter(model)
        self.save_plots = save_plots
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def reconstructions(self, x, x_rec, filename='recon.png', max_samples=8):
        """
        Visualize original vs reconstructed tabular sequences.
        """
        
        n_plot = min(max_samples, x.shape[0])
        fig, axs = plt.subplots(n_plot, x.shape[2], figsize=(x.shape[2] * 3, n_plot * 2))
        for i in range(n_plot):
            for j in range(x.shape[2]):
                axs[i, j].plot(x[i, :, j], label='Original', color='blue')
                axs[i, j].plot(x_rec[i, :, j], label='Reconstruction', color='orange', linestyle='--')
                if i == 0:
                    axs[i, j].set_title(f'Feature {j}')
                if j == 0:
                    axs[i, j].set_ylabel(f'Sample {i}')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()

        if self.save_plots:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def samples(self, generated, n_samples=8, filename='samples.png'):
        """
        Visualize generated samples from the prior.
        """
        
        n_plot = min(n_samples, generated.shape[0])
        samples = generated[:n_plot]
        fig, axs = plt.subplots(n_plot, samples.shape[2], figsize=(samples.shape[2] * 3, n_plot * 2))

        for i in range(n_plot):
            for j in range(samples.shape[2]):
                axs[i, j].plot(samples[i, :, j], label='Generated', color='green')
                if i == 0:
                    axs[i, j].set_title(f'Feature {j}')
                if j == 0:
                    axs[i, j].set_ylabel(f'Sample {i}')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()

        if self.save_plots:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def latent_traversal(self, cont_idx=None, disc_idx=None, steps=12, filename='latent_traversal.png'):
        """
        Traverse ONLY ONE latent dimension and observe output feature changes.
        """
        
        # Check: Only one of cont_idx or disc_idx should be set
        if (cont_idx is not None and disc_idx is not None) or (cont_idx is None and disc_idx is None):
            raise ValueError("You must specify exactly one of cont_idx or disc_idx.")
            
        # Check indexes don't exceed the number of latent variables
        if cont_idx is not None:
            if cont_idx >= self.latent_traverser.cont_dim or cont_idx < 0:
                raise IndexError(f"cont_idx {cont_idx} out of range (0 to {self.latent_traverser.cont_dim - 1})")
        elif disc_idx is not None:
            if disc_idx >= len(self.latent_traverser.disc_dims) or disc_idx < 0:
                raise IndexError(f"disc_idx {disc_idx} out of range (0 to {len(self.latent_traverser.disc_dims) - 1})")


        self.model.eval()
        latents = self.latent_traverser.traverse_line(cont_idx=cont_idx, disc_idx=disc_idx, size=steps)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            decoded = self.model.decoder(latents.to(device)).cpu().numpy()
        plt.figure(figsize=(10, 6))
        for i in range(decoded.shape[1]):
            plt.plot(range(steps), decoded[:, i], marker='o', label=f'Feature {i}')
        plt.tight_layout()
        plt.title(
        f"Latent Traversal on {'Continuous' if cont_idx is not None else 'Discrete'} Dim "
        f"{cont_idx if cont_idx is not None else disc_idx}")
        plt.xlabel('Traversal Step')
        plt.ylabel('Decoded Feature Value')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if self.save_plots:
            filename = f"latent_traversal_{'cont' if cont_idx is not None else 'disc'}_{cont_idx if cont_idx is not None else disc_idx}.png"
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
