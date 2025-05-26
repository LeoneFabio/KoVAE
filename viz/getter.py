import numpy as np
import torch
from scipy import stats
from utils.utils_model import get_device


class Getter():
    def __init__(self, model):
        self.model = model

    def get_reconstructed_data(self, train_loader, time, final_index):
        # Compute reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        with torch.no_grad():
            recon_data = []
            for data in train_loader:
                recon_data.append(self.model(data['inter'], time, final_index, isTraining=False)[0])
        recon_data = np.vstack(recon_data)
        return recon_data
    
    def get_original_data(self, train_loader):
        x = []
        for data in train_loader:
            x.append(data['original_data'].to(get_device()).float())
        x = np.vstack(x)
        return x
    
    def get_generated_data(self, train_loader):
        self.model.eval()
        with torch.no_grad():
            generated_data = []
            for data in train_loader:
                n_sample = data['original_data'].shape[0]
                generated_data.append(self.model.sample_data(n_sample, isTraining=False).detach().cpu().numpy())
        generated_data = np.vstack(generated_data)
        return generated_data

        