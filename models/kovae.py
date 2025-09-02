import torch
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

        ########################################################
        self.constraint_layer = EVPhysicalConstraintLayer()
        ########################################################

    def forward(self, z):
        # decode
        h, _ = self.rnn(z)
        x_hat = nn.functional.sigmoid(self.linear(h))

        ######################################################
        x_hat = self.constraint_layer(x_hat)
        ######################################################

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
        #################################################################self.decoder = VKDecoder(self.args, self.latent_dim)

        original_decoder = VKDecoder(self.args, self.latent_dim)
        self.decoder = modify_existing_decoder(original_decoder)

        #################################################################

    
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
        '''if a0 > 0:
            recon_loss = F.mse_loss(x_rec, x, reduction='sum') / batch_size
            loss += a0 * recon_loss
            agg_losses.append(recon_loss)'''
        physics_weight = getattr(self.args, 'physics_weight', 1.0)
        physics_loss_fn = EVLossWithPhysics(physics_weight=physics_weight)
        
        if a0 > 0:
            total_recon_loss, recon_loss, physics_loss = physics_loss_fn(x_rec, x)
            loss += a0 * total_recon_loss
            agg_losses.extend([recon_loss, physics_loss])  # for monitoring
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
        
class EVPhysicalConstraintLayer(nn.Module):
    """
    Post-processing layer to enforce EV physical constraints
    Can be added to existing decoder output without changing architecture
    """
    def __init__(self, feature_indices=None):
        super(EVPhysicalConstraintLayer, self).__init__()
        
        # Feature indices in your data: [odo, end_odo, soc, end_soc, event, charge_mode, duration]
        if feature_indices is None:
            self.odo_idx = 0
            self.end_odo_idx = 1 
            self.soc_idx = 2
            self.end_soc_idx = 3
            self.event_idx = 4
            self.charge_mode_idx = 5
            self.duration_idx = 6
        else:
            self.__dict__.update(feature_indices)
    
    def forward(self, x_raw):
        """
        Apply physical constraints to raw decoder output
        x_raw: [batch, seq_len, features] - raw decoder output
        Returns: [batch, seq_len, features] - physically constrained output
        """
        batch_size, seq_len, _ = x_raw.shape
        x_constrained = x_raw.clone()
        
        # Process each timestep sequentially to maintain temporal consistency
        for t in range(seq_len):
            if t == 0:
                # First timestep: apply basic range constraints only
                x_constrained[:, t] = self._apply_range_constraints(x_constrained[:, t])
            else:
                # Subsequent timesteps: apply temporal consistency constraints
                x_constrained[:, t] = self._apply_temporal_constraints(
                    x_constrained[:, t], 
                    x_constrained[:, t-1]
                )
        
        return x_constrained
    
    def _apply_range_constraints(self, x_t):
        """Apply basic range constraints for single timestep"""
        x_constrained = x_t.clone()
        
        # SOC constraints: 0-100% (assuming normalized to 0-1)
        x_constrained[:, self.soc_idx] = torch.clamp(x_constrained[:, self.soc_idx], 0.0, 1.0)
        x_constrained[:, self.end_soc_idx] = torch.clamp(x_constrained[:, self.end_soc_idx], 0.0, 1.0)
        
        # Event: binary (trip=0, charge=1) 
        x_constrained[:, self.event_idx] = torch.round(torch.sigmoid(x_constrained[:, self.event_idx]))
        
        # Charge mode: 0,1,2,3 (only meaningful when event=1)
        charge_mode_raw = x_constrained[:, self.charge_mode_idx]
        x_constrained[:, self.charge_mode_idx] = torch.clamp(torch.round(charge_mode_raw), 0, 3)
        
        # Duration: positive values
        x_constrained[:, self.duration_idx] = torch.relu(x_constrained[:, self.duration_idx])
        
        return x_constrained
    
    def _apply_temporal_constraints(self, x_t, x_prev):
        """Apply temporal consistency constraints between timesteps"""
        x_constrained = x_t.clone()
        
        # 1. ODOMETER MONOTONICITY
        prev_end_odo = x_prev[:, self.end_odo_idx]
        
        # Ensure odo[t] >= end_odo[t-1] (continuity)
        raw_odo = x_constrained[:, self.odo_idx]
        x_constrained[:, self.odo_idx] = torch.max(raw_odo, prev_end_odo)
        
        # Ensure end_odo[t] >= odo[t] (no backward travel)
        raw_end_odo = x_constrained[:, self.end_odo_idx]
        x_constrained[:, self.end_odo_idx] = torch.max(raw_end_odo, x_constrained[:, self.odo_idx])
        
        # 2. SOC CONTINUITY AND PHYSICS
        prev_end_soc = x_prev[:, self.end_soc_idx]
        
        # SOC[t] should start where previous ended (small tolerance for measurement error)
        measurement_noise = torch.randn_like(prev_end_soc) * 0.01  # 1% noise
        x_constrained[:, self.soc_idx] = torch.clamp(prev_end_soc + measurement_noise, 0.0, 1.0)
        
        # 3. EVENT-BASED SOC LOGIC
        event = torch.round(torch.sigmoid(x_constrained[:, self.event_idx]))
        x_constrained[:, self.event_idx] = event
        
        trip_mask = (event < 0.5)  # Trip event
        charge_mask = (event >= 0.5)  # Charge event
        
        # Calculate trip distance (normalized)
        trip_distance = x_constrained[:, self.end_odo_idx] - x_constrained[:, self.odo_idx]
        
        # SOC change logic
        start_soc = x_constrained[:, self.soc_idx]
        raw_end_soc = x_constrained[:, self.end_soc_idx]
        
        # For TRIPS: SOC decreases based on distance (consumption model)
        consumption_rate = 0.3  # 30% SOC per normalized distance unit
        trip_soc_decrease = trip_distance * consumption_rate
        min_soc = torch.full_like(start_soc, 0.05)  # min 5% SOC tensor
        trip_end_soc = torch.clamp(start_soc - trip_soc_decrease, min_soc, torch.ones_like(start_soc))
        
        # For CHARGING: SOC increases (limited by battery capacity and charging physics)
        charge_duration_norm = torch.clamp(x_constrained[:, self.duration_idx], 0, 1)
        max_charge_rate = 0.8  # max 80% SOC increase per normalized time unit
        charge_increase = charge_duration_norm * max_charge_rate
        charge_end_soc = torch.clamp(start_soc + charge_increase, start_soc, 1.0)
        
        # Apply event-specific SOC logic
        final_end_soc = torch.where(trip_mask, trip_end_soc, charge_end_soc)
        x_constrained[:, self.end_soc_idx] = final_end_soc
        
        # 4. CHARGE MODE LOGIC
        # Charge mode only meaningful during charging events
        charge_mode = torch.clamp(torch.round(x_constrained[:, self.charge_mode_idx]), 0, 3)
        x_constrained[:, self.charge_mode_idx] = torch.where(charge_mask, charge_mode, torch.zeros_like(charge_mode))
        
        # 5. DURATION CONSISTENCY
        # Duration should be reasonable for the distance/SOC change
        raw_duration = torch.relu(x_constrained[:, self.duration_idx])
        
        # For trips: duration proportional to distance
        trip_duration = trip_distance * 100  # scale factor for normalized time
        
        # For charging: duration proportional to SOC increase  
        soc_increase = final_end_soc - start_soc
        charge_duration = torch.where(soc_increase > 0, soc_increase * 200, torch.tensor(5.0))  # min 5 time units
        
        consistent_duration = torch.where(trip_mask, trip_duration, charge_duration)
        x_constrained[:, self.duration_idx] = consistent_duration
        
        return x_constrained


class EVLossWithPhysics(nn.Module):
    """
    Extended loss function that penalizes physical violations
    """
    def __init__(self, feature_indices=None, physics_weight=1.0):
        super(EVLossWithPhysics, self).__init__()
        self.physics_weight = physics_weight
        
        if feature_indices is None:
            self.odo_idx = 0
            self.end_odo_idx = 1 
            self.soc_idx = 2
            self.end_soc_idx = 3
            self.event_idx = 4
        else:
            self.__dict__.update(feature_indices)
    
    def forward(self, x_recon, x_target):
        """
        Compute loss with physical constraint penalties
        """
        batch_size = x_recon.size(0)
        
        # Standard reconstruction loss
        recon_loss = F.mse_loss(x_recon, x_target, reduction='mean')
        
        # Physical constraint violation penalties
        physics_loss = 0.0
        
        # 1. Odometer monotonicity penalty
        odo_violations = 0.0
        for t in range(1, x_recon.size(1)):
            # end_odo[t-1] should <= odo[t]
            prev_end_odo = x_recon[:, t-1, self.end_odo_idx]
            curr_odo = x_recon[:, t, self.odo_idx]
            violations = torch.relu(prev_end_odo - curr_odo)  # penalize backward travel
            odo_violations += torch.mean(violations)
            
            # odo[t] should <= end_odo[t] 
            curr_end_odo = x_recon[:, t, self.end_odo_idx]
            violations = torch.relu(curr_odo - curr_end_odo)  # penalize negative trips
            odo_violations += torch.mean(violations)
        
        physics_loss += odo_violations
        
        # 2. SOC continuity penalty
        soc_continuity_loss = 0.0
        for t in range(1, x_recon.size(1)):
            prev_end_soc = x_recon[:, t-1, self.end_soc_idx]
            curr_soc = x_recon[:, t, self.soc_idx]
            # Should be approximately equal (allow small measurement noise)
            continuity_error = torch.abs(prev_end_soc - curr_soc)
            soc_continuity_loss += torch.mean(continuity_error)
        
        physics_loss += soc_continuity_loss
        
        # 3. Event logic penalty
        event_logic_loss = 0.0
        for t in range(x_recon.size(1)):
            event = x_recon[:, t, self.event_idx]
            soc_change = x_recon[:, t, self.end_soc_idx] - x_recon[:, t, self.soc_idx]
            
            # During trips (event≈0), SOC should decrease or stay same
            trip_mask = event < 0.5
            trip_violations = torch.relu(soc_change[trip_mask])  # penalize SOC increase during trips
            if torch.any(trip_mask):
                event_logic_loss += torch.mean(trip_violations)
            
            # During charging (event≈1), SOC should increase
            charge_mask = event >= 0.5  
            charge_violations = torch.relu(-soc_change[charge_mask])  # penalize SOC decrease during charging
            if torch.any(charge_mask):
                event_logic_loss += torch.mean(charge_violations)
        
        physics_loss += event_logic_loss
        
        # Total loss
        total_loss = recon_loss + self.physics_weight * physics_loss
        
        return total_loss, recon_loss, physics_loss


# MINIMAL MODIFICATIONS TO YOUR EXISTING DECODER
def modify_existing_decoder(original_decoder):
    """
    Minimal modification to add constraint layer to existing decoder
    """
    class ConstrainedDecoder(nn.Module):
        def __init__(self, original_decoder):
            super(ConstrainedDecoder, self).__init__()
            self.original_decoder = original_decoder
            self.constraint_layer = EVPhysicalConstraintLayer()
            
        def forward(self, z):
            # Get raw output from original decoder
            x_raw = self.original_decoder(z)
            
            # Apply physical constraints
            x_constrained = self.constraint_layer(x_raw)
            
            return x_constrained
    
    return ConstrainedDecoder(original_decoder)