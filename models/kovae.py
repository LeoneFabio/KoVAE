"""import torch
import torch.nn as nn
import torch.nn.functional as F
import models.losses as losses
from models.neuralCDE import NeuralCDE
from models.modules import FinalTanh
from utils.utils import device_available

EPS = 1e-12

class VKEncoderIrregular(nn.Module):
    def __init__(self, args):
        super(VKEncoderIrregular, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm
        self.num_layers = self.args.num_layers

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        ode_func = FinalTanh(self.inp_dim, self.hidden_dim, self.hidden_dim, self.num_layers)
        self.emb = NeuralCDE(func=ode_func, input_channels=self.inp_dim,
                    hidden_channels=self.hidden_dim, output_channels=self.hidden_dim).to(args.device)
        self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=1, batch_first=True)


    def forward(self, time, train_coeffs, final_index):
        # encode
        h = self.emb(time, train_coeffs, final_index)
        h, _ = self.rnn(h)
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKEncoder(nn.Module):
    def __init__(self, args, num_layers=3):
        super(VKEncoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.rnn = nn.GRU(input_size=self.inp_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=args.num_layers, batch_first=True)

    def forward(self, x):
        # encode
        h, _ = self.rnn(x)  # b x seq_len x channels
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKDecoder(nn.Module):
    def __init__(self, args, latent_dim=16):
        super(VKDecoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim 
        self.latent_dim = latent_dim

        self.rnn = nn.GRU(input_size=self.latent_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=args.num_layers, batch_first=True)

        self.linear = nn.Linear(self.args.hidden_dim * 2, self.args.inp_dim)


    def forward(self, z):
        # decode
        h, _ = self.rnn(z)
        x_hat = nn.functional.sigmoid(self.linear(h))
        return x_hat


class KoVAE(nn.Module):
    def __init__(self, args, latent_spec=None, temperature=0.67):
        super(KoVAE, self).__init__()
        self.args = args
        self.z_dim = args.z_dim  # latent
        self.channels = args.inp_dim  # seq channel (multivariate features)
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.seq_len = args.seq_len
        self.pinv_solver = args.pinv_solver
        self.missing_value = args.missing_value
        
        '''
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.'''
        
        if not latent_spec:
            self.latent_spec = {'cont': self.z_dim}
        else:
            self.latent_spec = latent_spec
        self.temperature = temperature
        

        self.is_continuous = 'cont' in self.latent_spec
        self.is_discrete = 'disc' in self.latent_spec

        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
       
        
        
        # Define encoder 
        if self.missing_value > 0.:
            self.encoder = VKEncoderIrregular(self.args)
        else:
            self.encoder = VKEncoder(self.args)


        # Define decoder    
        self.decoder = VKDecoder(self.args, self.latent_dim)



    
        # Prior network: GRUCell outputs both cont and disc prior parameters
        self.z_prior_gru = nn.GRUCell(self.latent_dim, self.hidden_dim)
        if self.is_continuous:
            self.z_prior_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.z_prior_logvar = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            z_alphas = []
            for disc_dim in self.latent_spec['disc']:
                z_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.z_prior_alphas = nn.ModuleList(z_alphas)


        # ----- Posterior of sequence  -----
        # Encode parameters of latent distribution
        if self.is_continuous:
            self.z_mean = nn.Linear(self.hidden_dim * 2 , self.latent_cont_dim)
            self.z_logvar = nn.Linear(self.hidden_dim * 2, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            z_alphas = []
            for disc_dim in self.latent_spec['disc']:
                z_alphas.append(nn.Linear(self.hidden_dim * 2, disc_dim))
            self.z_alphas = nn.ModuleList(z_alphas)

        self.names = ['total', 'rec', 'kl', 'pred_prior']

    
    def forward(self, x, time=None, final_index=None):

        # ------------- ENCODING PART -------------
        if time is not None and final_index is not None:
            z = self.encoder(time, x, final_index)
        else:
            z = self.encoder(x)

        
        # Output parameters of latent distribution from hidden representation
        z_dist = {}
        if self.is_continuous:
            z_dist['cont'] = [self.z_mean(z), self.z_logvar(z)]

        if self.is_discrete:
            z_dist['disc'] = []
            for z_alpha in self.z_alphas:
                z_dist['disc'].append(F.softmax(z_alpha(z), dim=1))
        
        # Reparameterization trick
        z_post = self.reparameterize(z_dist, random_sampling=True)



        #  ------------- PRIOR PART -------------
        z_prior_dist, z_prior_sample = self.sample_prior(z.size(0), self.seq_len, random_sampling=True)


        x_rec = self.decoder(z_post)

        return x_rec, z_dist, z_prior_dist, z_prior_sample

    def compute_operator_and_pred(self, z):
        z_past, z_future = z[:, :-1], z[:, 1:]  # split latent

        # solve linear system (broadcast)
        if self.pinv_solver:
            Ct = torch.linalg.pinv(z_past.reshape(-1, self.latent_dim)) @ z_future.reshape(-1, self.latent_dim)

        else:
            # self.qr_solver
            Q, R = torch.linalg.qr(z_past.reshape(-1, self.latent_dim))
            B = Q.T @ z_future.reshape(-1, self.latent_dim)
            Ct = torch.linalg.solve_triangular(R, B, upper=True)

        # predict (broadcast)
        z_pred = z_past @ Ct

        err = .0
        z_hat = z_past
        for jj in range(self.args.num_steps):
            z_hat = z_hat @ Ct
            err += (F.mse_loss(z_hat[:, :-jj or None], z[:, (jj + 1):]) / torch.norm(z_hat[:, :-jj or None], p='fro'))

        return Ct, z_pred, err

    def loss(self, x, x_rec, z_dist, z_prior_dist, z_prior_sample):
        '''
        :param x: Original input sequence
        :param x_rec: Reconstructed sequence
        :param z_dist: Posterior latent distributions (dict with 'cont' and/or 'disc')
        :param z_prior_dist: Prior latent distributions (same format)
        :param z_prior_sample: Prior-sampled full latent trajectory
        :return: tuple of (total loss, rec loss, KL loss, predictive loss)
        '''

        a0 = self.args.w_rec
        a1 = self.args.w_kl
        a2 = self.args.w_pred_prior
        batch_size = x.size(0)

        loss = 0.0
        agg_losses = []            


        # --- 1. Reconstruction Loss ---
        if a0 > 0:
            recon_loss = F.mse_loss(x_rec, x, reduction='sum') / batch_size
            loss += a0 * recon_loss
            agg_losses.append(recon_loss)
        else:
            recon_loss = torch.tensor(0.0, device=x.device)

        # --- 2. KL Divergence Loss ---
        kl_loss = torch.tensor(0.0, device=x.device)

        # Continuous KL
        if self.is_continuous and z_dist.get('cont') is not None:
            z_post_mean, z_post_logvar = z_dist['cont']
            z_prior_mean, z_prior_logvar = z_prior_dist['cont']

    
            kl_cont = losses.kl_normal_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            kl_cont = torch.sum(kl_cont) / batch_size
            kl_loss += kl_cont

        # Discrete KL
        if self.is_discrete and z_dist.get('disc') is not None and z_prior_dist.get('disc') is not None:
            for post_logit, prior_logit in zip(z_dist['disc'], z_prior_dist['disc']):
                # Posterior uses softmax over logits (already in z_dist)
                kl_disc = losses.kl_categorical_loss(post_logit, prior_logit)
                kl_loss += kl_disc

        if a1 > 0:
            loss += a1 * kl_loss
        agg_losses.append(kl_loss)

        # --- 3. Predictive Loss on Latent Prior ---
        if a2 > 0:
            # Check for valid latent prior sample
            if not torch.isnan(z_prior_sample[0]).any():
                _, _, pred_err_prior = self.compute_operator_and_pred(z_prior_sample)
                loss += a2 * pred_err_prior
            else:
                pred_err_prior = torch.tensor(0.0, device=x.device)
        else:
            pred_err_prior = torch.tensor(0.0, device=x.device)

        agg_losses.append(pred_err_prior)
        

        # Total loss first
        agg_losses = [loss] + agg_losses
        return tuple(agg_losses)



    def sample_data(self, n_sample):
        # sample from prior
        _, z_out = self.sample_prior(n_sample, self.seq_len, random_sampling=True)
        x_rec = self.decoder(z_out)
        return x_rec

    # ------ sample z purely from learned LSTM prior with arbitrary seq ------
    def sample_prior(self, n_sample, seq_len, random_sampling=True):
        device = device_available()

        if random_sampling:
            # Initialize with random noise instead of zeros
            z_t = torch.randn(n_sample, self.latent_dim, device=device) * 0.1
            h_t = torch.randn(n_sample, self.hidden_dim, device=device) * 0.1
        else:
            z_t = torch.zeros(n_sample, self.latent_dim, device=device)
            h_t = torch.zeros(n_sample, self.hidden_dim, device=device)

        z_seq = []
        cont_means, cont_logvars = [], []

        disc_logits = [[] for _ in range(self.num_disc_latents)]  # Track logits over time

        for _ in range(seq_len):
            h_t = self.z_prior_gru(z_t, h_t)
            
            z_parts = []

            if self.is_continuous:
                mean_t = self.z_prior_mean(h_t)
                logvar_t = self.z_prior_logvar(h_t)
                cont_sample = self.sample_normal(mean_t, logvar_t, random_sampling=random_sampling)

                cont_means.append(mean_t)
                cont_logvars.append(logvar_t)
                z_parts.append(cont_sample)

            if self.is_discrete:
                for i, alpha_layer in enumerate(self.z_prior_alphas):
                    logits = alpha_layer(h_t)  # raw logits
                    disc_logits[i].append(logits)
                    alpha = F.softmax(logits, dim=1)
                    disc_sample = self.sample_gumbel_softmax(alpha, random_sampling=random_sampling)

                    z_parts.append(disc_sample)
                    
            z_t = torch.cat(z_parts, dim=1)
           
            z_seq.append(z_t)

        z_seq = torch.stack(z_seq, dim=1)  # (B, T, latent_dim)

        latent_dist = {}
        if self.is_continuous:
            latent_dist['cont'] = [
                torch.stack(cont_means, dim=1),     # (B, T, latent_cont_dim)
                torch.stack(cont_logvars, dim=1),   # (B, T, latent_cont_dim)
            ]

        if self.is_discrete:
            # Stack each group of logits across time steps
            disc_logit_stacks = [torch.stack(logits_per_cat, dim=1) for logits_per_cat in disc_logits]
            latent_dist['disc'] = disc_logit_stacks  # list of (B, T, latent_dsc_dim)

        return latent_dist, z_seq

    
    
    def reparameterize(self, latent_dist, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            latent_sample = []
            if self.is_continuous:
                mean, logvar = latent_dist['cont']
                cont_sample = self.sample_normal(mean, logvar, random_sampling=random_sampling)
                latent_sample.append(cont_sample)

            if self.is_discrete:
                for alpha in latent_dist['disc']:
                    disc_sample = self.sample_gumbel_softmax(alpha, random_sampling=random_sampling)
                    latent_sample.append(disc_sample)            
            
            # Concatenate continuous and discrete samples into one large sample
            return torch.cat(latent_sample, dim=2)
        else:
            mean, _ = latent_dist['cont']
            return mean
        
    def sample_normal(self, mean, logvar, random_sampling):
        # Sample from a normal distribution
        if random_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if torch.cuda.is_available():
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha, random_sampling):    
        if random_sampling:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if torch.cuda.is_available():
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            max_alpha = torch.argmax(alpha, dim=-1) 
            one_hot_samples = torch.zeros_like(alpha)
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(-1, max_alpha.unsqueeze(-1), 1)
            if torch.cuda.is_available():
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples
        
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class SequentialEVVAE(nn.Module):
    """
    VAE with explicit temporal consistency constraints for EV data
    Ensures: monotonic odometer, consistent SOC transitions, logical event sequences
    """
    
    def __init__(self, args):
        super(SequentialEVVAE, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.inp_dim = args.inp_dim  # [odo, end_odo, soc, end_soc, event, charge_mode, duration]
        self.hidden_dim = args.hidden_dim
        self.z_dim = args.z_dim
        
        # Feature indices (based on your EV data structure)
        self.odo_idx = 0
        self.end_odo_idx = 1
        self.soc_idx = 2
        self.end_soc_idx = 3
        self.event_idx = 4
        self.charge_mode_idx = 5
        self.duration_idx = 6
        
        # Encoder
        self.encoder = nn.GRU(
            input_size=self.inp_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dim * 2, self.z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim * 2, self.z_dim)
        
        # Decoder with temporal awareness
        self.decoder_gru = nn.GRU(
            input_size=self.z_dim + self.inp_dim,  # latent + previous state
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output layers for each feature with constraints
        self.odo_predictor = nn.Linear(self.hidden_dim, 1)
        self.soc_predictor = nn.Linear(self.hidden_dim, 1)
        self.event_predictor = nn.Linear(self.hidden_dim, 2)  # binary: trip/charge
        self.charge_mode_predictor = nn.Linear(self.hidden_dim, 4)  # 0,1,2,3
        self.duration_predictor = nn.Linear(self.hidden_dim, 1)
        
    def encode(self, x):
        """Encode sequence to latent space"""
        h, _ = self.encoder(x)
        # Use final hidden state
        h_final = h[:, -1, :]
        
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_with_constraints(self, z, initial_state=None):
        """
        Decode with temporal consistency constraints
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize sequence
        if initial_state is None:
            # Start with reasonable initial values
            current_state = torch.zeros(batch_size, self.inp_dim, device=device)
            current_state[:, self.odo_idx] = 0.5  # normalized initial odometer
            current_state[:, self.soc_idx] = 0.8   # start with high SOC
            current_state[:, self.end_soc_idx] = 0.8
        else:
            current_state = initial_state.clone()
        
        sequence = []
        hidden = None
        
        # Expand z to sequence length
        z_expanded = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        for t in range(self.seq_len):
            # Concatenate latent code with previous state
            decoder_input = torch.cat([z_expanded[:, t:t+1], current_state.unsqueeze(1)], dim=-1)
            
            # GRU step
            output, hidden = self.decoder_gru(decoder_input, hidden)
            h_t = output.squeeze(1)
            
            # Predict next state with constraints
            next_state = self.apply_constraints(h_t, current_state)
            
            sequence.append(next_state)
            current_state = next_state
        
        return torch.stack(sequence, dim=1)
    
    def apply_constraints(self, h_t, prev_state):
        """
        Apply physical and logical constraints to predictions
        """
        batch_size = h_t.size(0)
        device = h_t.device
        next_state = torch.zeros(batch_size, self.inp_dim, device=device)
        
        # 1. Odometer constraint: monotonically increasing
        prev_end_odo = prev_state[:, self.end_odo_idx]
        odo_increment = torch.relu(self.odo_predictor(h_t).squeeze())  # positive increment
        next_odo = prev_end_odo + odo_increment * 0.1  # scale increment
        
        # 2. Trip distance (end_odo - odo) should be reasonable (0-100km normalized)
        trip_distance = torch.sigmoid(self.duration_predictor(h_t).squeeze()) * 0.2  # max 20% of range
        next_end_odo = next_odo + trip_distance
        
        # 3. SOC constraints: starts where previous ended, decreases during trips
        prev_end_soc = prev_state[:, self.end_soc_idx]
        next_soc = prev_end_soc  # continuity
        
        # 4. Event prediction: trip (0) or charge (1)
        event_logits = self.event_predictor(h_t)
        event_probs = torch.softmax(event_logits, dim=1)
        event = torch.multinomial(event_probs, 1).squeeze().float()
        
        # 5. SOC evolution based on event
        if torch.any(event == 0):  # Trip
            # SOC decreases during trips (proportional to distance)
            trip_mask = (event == 0)
            soc_decrease = trip_distance * 0.5  # consumption rate
            next_end_soc = torch.where(trip_mask, 
                                     torch.clamp(next_soc - soc_decrease, 0.1, 1.0),
                                     next_soc)
            charge_mode = torch.zeros_like(event)
        else:  # Charge
            # SOC increases during charging
            charge_mask = (event == 1)
            soc_increase = torch.sigmoid(self.soc_predictor(h_t).squeeze()) * 0.5
            next_end_soc = torch.where(charge_mask,
                                     torch.clamp(next_soc + soc_increase, 0.1, 1.0),
                                     next_soc)
            # Predict charge mode for charging events
            charge_mode_logits = self.charge_mode_predictor(h_t)
            charge_mode = torch.multinomial(torch.softmax(charge_mode_logits, dim=1), 1).squeeze().float()
            charge_mode = torch.where(charge_mask, charge_mode, torch.zeros_like(charge_mode))
        
        # 6. Duration prediction (reasonable values)
        duration_raw = self.duration_predictor(h_t).squeeze()
        duration = torch.sigmoid(duration_raw) * 500  # 0-500 minutes normalized
        
        # Assemble next state
        next_state[:, self.odo_idx] = next_odo
        next_state[:, self.end_odo_idx] = next_end_odo
        next_state[:, self.soc_idx] = next_soc
        next_state[:, self.end_soc_idx] = next_end_soc
        next_state[:, self.event_idx] = event
        next_state[:, self.charge_mode_idx] = charge_mode
        next_state[:, self.duration_idx] = duration / 500.0  # normalize
        
        return next_state
    
    def forward(self, x):
        """Forward pass"""
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode with constraints
        x_recon = self.decode_with_constraints(z, x[:, 0])  # start from first state
        
        return x_recon, mu, logvar
    
    def generate_sequence(self, n_samples=1, seq_len=None):
        """Generate new sequences with proper temporal consistency"""
        if seq_len is None:
            seq_len = self.seq_len
            
        device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(n_samples, self.z_dim, device=device)
        
        # Generate with constraints
        generated = self.decode_with_constraints(z)
        
        return generated
    
    def loss_function(self, x_recon, x, mu, logvar):
        """
        Loss with temporal consistency penalties
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. Temporal consistency losses
        consistency_loss = 0.0
        
        # Odometer monotonicity
        odo_diff = x_recon[:, 1:, self.odo_idx] - x_recon[:, :-1, self.end_odo_idx]
        monotonic_penalty = torch.sum(torch.relu(-odo_diff))  # penalize decreases
        consistency_loss += monotonic_penalty / batch_size
        
        # SOC continuity (end_soc[t] should equal soc[t+1])
        soc_continuity = torch.sum((x_recon[:, 1:, self.soc_idx] - 
                                   x_recon[:, :-1, self.end_soc_idx]).pow(2))
        consistency_loss += soc_continuity / batch_size
        
        # Event logic: SOC should increase during charging, decrease during trips
        event_mask = x_recon[:, :, self.event_idx] > 0.5  # charging
        soc_change = x_recon[:, :, self.end_soc_idx] - x_recon[:, :, self.soc_idx]
        
        # Charging should increase SOC
        charge_logic_loss = torch.sum(torch.relu(-soc_change[event_mask]))
        # Trips should decrease SOC (or stay same for very short trips)
        trip_logic_loss = torch.sum(torch.relu(soc_change[~event_mask] - 0.01))
        
        consistency_loss += (charge_logic_loss + trip_logic_loss) / batch_size
        
        # Total loss
        total_loss = recon_loss + self.args.w_kl * kl_loss + self.args.w_consistency * consistency_loss
        
        return total_loss, recon_loss, kl_loss, consistency_loss


class SequentialTrainer:
    """Training loop with validation of temporal constraints"""
    
    def __init__(self, model, train_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.args = args
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            x = data['data'][:, :, :-1]  # remove time column
            
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(x)
            
            loss, recon_loss, kl_loss, consistency_loss = self.model.loss_function(
                x_recon, x, mu, logvar
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss={loss.item():.4f}, '
                      f'Recon={recon_loss.item():.4f}, '
                      f'KL={kl_loss.item():.4f}, '
                      f'Consistency={consistency_loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate_temporal_consistency(self, generated_data):
        """Validate that generated data follows temporal rules"""
        batch_size, seq_len, _ = generated_data.shape
        violations = {}
        
        # Check odometer monotonicity
        odo_violations = 0
        for i in range(seq_len - 1):
            end_odo_current = generated_data[:, i, 1]
            odo_next = generated_data[:, i + 1, 0]
            violations_batch = torch.sum(odo_next < end_odo_current).item()
            odo_violations += violations_batch
        
        violations['odometer_monotonic'] = odo_violations / (batch_size * (seq_len - 1))
        
        # Check SOC continuity
        soc_violations = 0
        for i in range(seq_len - 1):
            end_soc_current = generated_data[:, i, 3]
            soc_next = generated_data[:, i + 1, 2]
            violations_batch = torch.sum(torch.abs(soc_next - end_soc_current) > 0.05).item()
            soc_violations += violations_batch
        
        violations['soc_continuity'] = soc_violations / (batch_size * (seq_len - 1))
        
        return violations
    
    def generate_and_validate(self, n_samples=10):
        """Generate samples and validate temporal consistency"""
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate_sequence(n_samples)
            violations = self.validate_temporal_consistency(generated)
            
        return generated, violations


# Usage example
def train_sequential_ev_vae(args, train_loader):
    """
    Complete training pipeline for sequential EV VAE
    """
    # Add consistency weight to args if not present
    if not hasattr(args, 'w_consistency'):
        args.w_consistency = 1.0
    
    # Initialize model
    model = SequentialEVVAE(args)
    trainer = SequentialTrainer(model, train_loader, args)
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch()
        
        # Generate and validate every 10 epochs
        if epoch % 10 == 0:
            generated, violations = trainer.generate_and_validate()
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}')
            print(f'Validation - Odometer violations: {violations["odometer_monotonic"]:.3f}')
            print(f'Validation - SOC violations: {violations["soc_continuity"]:.3f}')
    
    return model, trainer

    