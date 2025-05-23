import torch
import torch.nn.functional as F
import numpy as np


def kl_normal_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    # COMPUTE KL DIV
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * (z_prior_logvar - z_post_logvar +
                   ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return kld_z

def kl_categorical_loss(post_logits, prior_logit, eps=1e-10):
    """
    KL divergence between categorical (softmax) posterior and unifrm prior probabilities.
    logits: Tensor of shape (B, T, D)
    """
    q = F.softmax(post_logits, dim=-1) + eps
    p = F.softmax(prior_logit, dim=-1) + eps
    log_q = torch.log(q)
    log_p = torch.log(p)
    kl = torch.sum(q * (log_q - log_p), dim=-1)  # (B, T)
    return kl.mean()


def eig_loss(Ct_prior, mode):
    # COMPUTE EIG PENALTY:
    eigs = torch.abs(torch.linalg.eigvals(Ct_prior))
    if mode == '2':
        eigs_latent = torch.argsort(eigs)[-2:]
        one_valued_eig = torch.ones_like(eigs[eigs_latent], device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs[eigs_latent], one_valued_eig)
    if mode == '3':
        eigs_latent = torch.argsort(eigs)[-3:]
        one_valued_eig = torch.ones_like(eigs[eigs_latent], device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs[eigs_latent], one_valued_eig)
    if mode == '4':
        one_valued_eig = torch.ones_like(eigs, device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs, one_valued_eig)

    return eig_loss_prior


def eigen_constraints(Ct_post, Ct_prior):
    eig_post = torch.abs(torch.linalg.eigvals(Ct_post))
    eig_prior = torch.abs(torch.linalg.eigvals(Ct_prior))
    eig_post_sorted = torch.argsort(eig_post, descending=True)
    eig_prior_sorted = torch.argsort(eig_prior, descending=True)
    eig_norm_one = .5
    eigs_to_one = int(eig_norm_one * len(eig_post_sorted))
    eigs_to_one_post, eigs_less_than_one_post = eig_post[eig_post_sorted[:eigs_to_one]], \
        eig_post[eig_post_sorted[eigs_to_one:]]
    eigs_to_one_prior, eigs_less_than_one_prior = eig_prior[eig_prior_sorted[:eigs_to_one]], \
        eig_prior[eig_prior_sorted[eigs_to_one:]]
    one_valued_eig = torch.ones_like(eigs_to_one_post)
    return eigs_less_than_one_post, eigs_less_than_one_prior, eigs_to_one_post, eigs_to_one_prior, one_valued_eig

