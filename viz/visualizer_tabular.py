import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from viz.latent_traversals import LatentTraverser


class TabularVisualizer:
    def __init__(self, model, output_dir='./visualizations', save_plots=True):
        """
        Visualizer for KoVAE applied to tabular time series data.
        """
        self.model = model
        self.latent_traverser = LatentTraverser(model.latent_spec)
        self.save_plots = save_plots
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def reconstructions(self, data, filename='recon.png', max_samples=8):
        """
        Visualize original vs reconstructed tabular sequences.
        """
        self.model.eval()
        x = data['data'].to(next(self.model.parameters()).device).float()
        x = x[:, :, :-1]
        coeffs = data['inter']
        time = torch.FloatTensor(list(range(x.shape[1]))).to(x.device)
        final_index = (torch.ones(x.shape[0]) * (x.shape[1] - 1)).to(x.device).float()

        with torch.no_grad():
            x_rec = self.model(coeffs, time, final_index)[0]

        x = x.cpu().numpy()
        x_rec = x_rec.cpu().numpy()

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

    def samples(self, n_samples=7, filename='samples.png'):
        """
        Generate new samples from the prior.
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample_data(n_samples).cpu().numpy()

        fig, axs = plt.subplots(n_samples, samples.shape[2], figsize=(samples.shape[2] * 3, n_samples * 2))

        for i in range(n_samples):
            for j in range(samples.shape[2]):
                axs[i, j].plot(samples[i, :, j], color='green')
                if i == 0:
                    axs[i, j].set_title(f'Feature {j}')
        plt.tight_layout()

        if self.save_plots:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def latent_traversal(self, cont_idx=0, steps=8, filename='latent_traversal.png'):
        """
        Traverse one latent dimension and observe output feature changes.
        """
        self.model.eval()
        latents = self.latent_traverser.traverse_line(cont_idx=cont_idx, size=steps)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            decoded = self.model.decoder(latents.to(device)).cpu().numpy()
        plt.figure(figsize=(10, 6))
        for i in range(decoded.shape[1]):
            plt.plot(range(steps), decoded[:, i], marker='o', label=f'Feature {i}')
        plt.tight_layout()
        plt.title(f'Latent Traversal on Continuous Dim {cont_idx}')
        plt.xlabel('Traversal Step')
        plt.ylabel('Decoded Feature Value')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if self.save_plots:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
