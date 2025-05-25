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
        ############################################################################
        
        
        # Define encoder 
        if self.missing_value > 0.:
            self.encoder = VKEncoderIrregular(self.args)
        else:
            self.encoder = VKEncoder(self.args)


        # Define decoder    
        self.decoder = VKDecoder(self.args, self.latent_dim)



        ##############################################################################
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

    
    def forward(self, x, time=None, final_index=None, isTraining=True):

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
        z_post = self.reparameterize(z_dist, random_sampling=True, isTraining=isTraining)

        '''Z_enc = {
            'mean': latent_dist.get('cont', [None])[0],
            'logvar': latent_dist.get('cont', [None, None])[1],
            'sample': z_post
        }'''


        #  ------------- PRIOR PART -------------
        z_prior_dist, z_prior_sample = self.sample_prior(z.size(0), self.seq_len, random_sampling=True, isTraining=isTraining)
        
        '''Z_enc_prior = {
            'mean': z_prior_dist.get('cont', [None])[0],
            'logvar': z_prior_dist.get('cont', [None, None])[1],
            'sample': z_prior_sample
        }'''

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
        """
        :param x: Original input sequence
        :param x_rec: Reconstructed sequence
        :param z_dist: Posterior latent distributions (dict with 'cont' and/or 'disc')
        :param z_prior_dist: Prior latent distributions (same format)
        :param z_prior_sample: Prior-sampled full latent trajectory
        :return: tuple of (total loss, rec loss, KL loss, predictive loss)
        """

        a0 = self.args.w_rec
        a1 = self.args.w_kl
        a2 = self.args.w_pred_prior
        batch_size = x.size(0)

        loss = 0.0
        agg_losses = []
        
        
        print("x min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
        print("x_rec min/max/mean:", x_rec.min().item(), x_rec.max().item(), x_rec.mean().item())

        if torch.isnan(x).any():
            print("[NaN WARNING] x contains NaNs")
        if torch.isnan(x_rec).any():
            print("[NaN WARNING] x_rec contains NaNs")
            


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
            
            print("z_post_mean min/max/mean:", z_post_mean.min().item(), z_post_mean.max().item(), z_post_mean.mean().item())
            print("z_post_logvar min/max/mean:", z_post_logvar.min().item(), z_post_logvar.max().item(), z_post_logvar.mean().item())

            if torch.isnan(z_post_mean).any():
                print("[NaN WARNING] z_post_mean has NaNs")
            if torch.isnan(z_post_logvar).any():
                print("[NaN WARNING] z_post_logvar has NaNs")

    
            kl_cont = losses.kl_normal_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            kl_cont = torch.sum(kl_cont) / batch_size
            kl_loss += kl_cont

        # Discrete KL
        if self.is_discrete and z_dist.get('disc') is not None and z_prior_dist.get('disc') is not None:
            for post_logit, prior_logit in zip(z_dist['disc'], z_prior_dist['disc']):
                print("post_logit min/max/mean:", post_logit.min().item(), post_logit.max().item(), post_logit.mean().item())
                print("prior_logit min/max/mean:", prior_logit.min().item(), prior_logit.max().item(), prior_logit.mean().item())

                if torch.isnan(post_logit).any():
                    print("[NaN WARNING] post_logit has NaNs")
                if torch.isnan(prior_logit).any():
                    print("[NaN WARNING] prior_logit has NaNs")
                # Posterior uses softmax over logits (already in z_dist)
                kl_disc = losses.kl_categorical_loss(post_logit, prior_logit)
                kl_loss += kl_disc

        if a1 > 0:
            loss += a1 * kl_loss
        agg_losses.append(kl_loss)

        # --- 3. Predictive Loss on Latent Prior ---
        if a2 > 0:
            print("z_prior_sample[0] min/max/mean:", z_prior_sample[0].min().item(), z_prior_sample[0].max().item(), z_prior_sample[0].mean().item())
        
        if torch.isnan(z_prior_sample[0]).any():
            print("[NaN WARNING] z_prior_sample[0] has NaNs")

            _, _, pred_err_prior = self.compute_operator_and_pred(z_prior_sample)
            loss += a2 * pred_err_prior
        else:
            pred_err_prior = torch.tensor(0.0, device=x.device)

        agg_losses.append(pred_err_prior)
        
        '''# === DEBUG PRINTS ===
        print("=== LOSS BREAKDOWN ===")
        print("Reconstruction loss:", recon_loss.item())
        print("KL continuous:", kl_cont.item())
        print("KL discrete:", kl_disc.item())
        print("KL total:", kl_loss.item())
        print("Pred error prior:", pred_err_prior.item())
        print("Total loss:", loss.item())

        # Check for NaNs/Infs
        for name, val in [("recon_loss", recon_loss), ("kl_cont", kl_cont),
                          ("kl_disc", kl_disc), ("pred_err_prior", pred_err_prior), ("loss", loss)]:
            if torch.isnan(val).any():
                print(f"[NaN WARNING] {name} contains NaN")
            if torch.isinf(val).any():
                print(f"[Inf WARNING] {name} contains Inf")'''
        

        # Total loss first
        agg_losses = [loss] + agg_losses
        return tuple(agg_losses)



    def sample_data(self, n_sample, isTraining=False):
        # sample from prior
        _, z_out = self.sample_prior(n_sample, self.seq_len, random_sampling=True, isTraining=isTraining)
        x_rec = self.decoder(z_out)
        return x_rec

    # ------ sample z purely from learned LSTM prior with arbitrary seq ------
    def sample_prior(self, n_sample, seq_len, random_sampling=True, isTraining=False):
        device = device_available()

        z_t = torch.zeros(n_sample, self.latent_dim, device=device)
        h_t = torch.zeros(n_sample, self.hidden_dim, device=device)

        z_seq = []
        cont_means, cont_logvars = [], []

        disc_logits = [[] for _ in range(self.num_disc_latents)]  # Track logits over time

        for _ in range(seq_len):
            
            print("z_t before GRU min/max/mean:", z_t.min().item(), z_t.max().item(), z_t.mean().item())
            if torch.isnan(z_t).any():
                print("[NaN WARNING] z_t has NaNs BEFORE GRU")

            h_t = self.z_prior_gru(z_t, h_t)
            
            print("z_prior loop -- h_t min/max/mean:", h_t.min().item(), h_t.max().item(), h_t.mean().item())
            if torch.isnan(h_t).any():
                print("[NaN WARNING] h_t has NaNs")


            z_parts = []

            if self.is_continuous:
                mean_t = self.z_prior_mean(h_t)
                logvar_t = self.z_prior_logvar(h_t)
                cont_sample = self.sample_normal(mean_t, logvar_t, isTraining)

                cont_means.append(mean_t)
                cont_logvars.append(logvar_t)
                z_parts.append(cont_sample)

            if self.is_discrete:
                for i, alpha_layer in enumerate(self.z_prior_alphas):
                    logits = alpha_layer(h_t)  # raw logits
                    disc_logits[i].append(logits)
                    alpha = F.softmax(logits, dim=1)
                    disc_sample = self.sample_gumbel_softmax(alpha, isTraining)
                    

                    z_parts.append(disc_sample)
            print("z_t END OF THE ITERATION min/max/mean")
            print(z_parts)
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

    
    
    def reparameterize(self, latent_dist, random_sampling=True, isTraining=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            latent_sample = []
            if self.is_continuous:
                mean, logvar = latent_dist['cont']
                cont_sample = self.sample_normal(mean, logvar, isTraining)
                latent_sample.append(cont_sample)

            if self.is_discrete:
                for alpha in latent_dist['disc']:
                    disc_sample = self.sample_gumbel_softmax(alpha, isTraining)
                    latent_sample.append(disc_sample)            
            
            # Concatenate continuous and discrete samples into one large sample
            return torch.cat(latent_sample, dim=2)
        else:
            mean, _ = latent_dist['cont']
            return mean
        
    def sample_normal(self, mean, logvar, isTraining):
        # Sample from a normal distribution
        if isTraining:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if torch.cuda.is_available():
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean
    
    def sample_gumbel_softmax(self, alpha, isTraining):    
        if isTraining:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if torch.cuda.is_available():
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            print(" ----------------------- LOGITS ------------------------- ")
            print(logit)
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            max_alpha = torch.argmax(alpha, dim=-1) 
            one_hot_samples = torch.zeros_like(alpha)
            print("one_hot_samples shape:")
            print(one_hot_samples.shape)
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(-1, max_alpha.unsqueeze(-1), 1)
            if torch.cuda.is_available():
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples
        
    
    