import torch
import torch.nn as nn
import torch.nn.functional as F

class EVDomainPenalties:
    """
    Domain-specific penalty calculator for EV charging/driving patterns
    """
    
    @staticmethod
    def compute_trip_soc_penalty(x_rec, event_idx=0, charge_idx=4, threshold=0.5):
        """
        Penalize trips where ending SoC > starting SoC
        
        Args:
            x_rec: Reconstructed sequence (B, T, C) where C includes [event, charge_mode, duration, km, charge]
            event_idx: Index of event type in channel dimension
            charge_idx: Index of charge column in channel dimension
            threshold: Threshold to classify as "trip" event (assuming sigmoid output)
        
        Returns:
            penalty: Scalar penalty value
        """
        batch_size, seq_len, channels = x_rec.shape
        
        # Identify trip events (assuming event is binary: 0=trip, 1=charge after sigmoid)
        # If event < threshold, it's likely a trip
        is_trip = x_rec[:, :, event_idx] < threshold  # (B, T)
        
        # Get charge deltas
        charge_delta = x_rec[:, :, charge_idx]  # (B, T)
        
        # Penalize positive charge during trips (trips should discharge)
        trip_positive_charge = torch.relu(charge_delta) * is_trip.float()
        
        penalty = torch.sum(trip_positive_charge) / batch_size
        return penalty
    
    @staticmethod
    def compute_consecutive_charge_penalty(x_rec, event_idx=0, charge_threshold=0.5, 
                                          charge_idx=4, min_charge_delta=0.01):
        """
        Penalize multiple consecutive charging events
        
        Args:
            x_rec: Reconstructed sequence (B, T, C)
            event_idx: Index of event type
            charge_threshold: Threshold to classify as "charge" event
            charge_idx: Index of charge column
            min_charge_delta: Minimum charge change to consider as actual charging
        
        Returns:
            penalty: Scalar penalty value
        """
        batch_size, seq_len, channels = x_rec.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=x_rec.device)
        
        # Identify charging events
        '''is_charge = x_rec[:, :, event_idx] >= charge_threshold  # (B, T)
        has_charge = x_rec[:, :, charge_idx] > min_charge_delta  # Actually adding charge
        
        is_charging_event = (is_charge & has_charge).float()  # (B, T)'''
        
        
        is_charge = torch.sigmoid(10 * (x_rec[:, :, event_idx] - charge_threshold))  # smooth ">= threshold"
        has_charge = torch.sigmoid(10 * (x_rec[:, :, charge_idx] - min_charge_delta))
        is_charging_event = is_charge * has_charge  # smooth AND

        print("x_rec shape:", x_rec.shape)
        print("event_idx:", event_idx, "charge_idx:", charge_idx)
        print("is_charge:", is_charge.mean().item(), "has_charge:", has_charge.mean().item())

        # === DEBUG INFO ===
    
        print("\n[DEBUG: Sigmoid smoothing check]")
        print(f"  x_rec shape: {tuple(x_rec.shape)}")
        print(f"  event_idx={event_idx}, charge_idx={charge_idx}")
        print(f"  charge_threshold={charge_threshold}, min_charge_delta={min_charge_delta}")
    
        # Ranges of inputs to the sigmoid
        evt_vals = x_rec[:, :, event_idx]
        chg_vals = x_rec[:, :, charge_idx]
    
        print(f"  event value stats: min={evt_vals.min().item():.4f}, max={evt_vals.max().item():.4f}, mean={evt_vals.mean().item():.4f}")
        print(f"  charge value stats: min={chg_vals.min().item():.4f}, max={chg_vals.max().item():.4f}, mean={chg_vals.mean().item():.4f}")
    
        # Check smooth activation ranges
        print(f"  is_charge (sigmoid output): min={is_charge.min().item():.4f}, max={is_charge.max().item():.4f}, mean={is_charge.mean().item():.4f}")
        print(f"  has_charge (sigmoid output): min={has_charge.min().item():.4f}, max={has_charge.max().item():.4f}, mean={has_charge.mean().item():.4f}")
        print(f"  is_charging_event (combined): min={is_charging_event.min().item():.4f}, max={is_charging_event.max().item():.4f}, mean={is_charging_event.mean().item():.4f}")
    
        # Check if the sigmoid saturates (values too close to 0 or 1)
        saturate_is_charge = ((is_charge < 0.01) | (is_charge > 0.99)).float().mean().item()
        saturate_has_charge = ((has_charge < 0.01) | (has_charge > 0.99)).float().mean().item()
        print(f" Saturation ratio: is_charge={saturate_is_charge:.3f}, has_charge={saturate_has_charge:.3f}")
    
        # Check gradient connectivity
        print(f"  requires_grad: x_rec={x_rec.requires_grad}, is_charge={is_charge.requires_grad}, is_charging_event={is_charging_event.requires_grad}")
        print(f"  grad_fn: is_charge={is_charge.grad_fn}, is_charging_event={is_charging_event.grad_fn}")
    
        print("  Sigmoid smoothing debug complete.\n")
    

        # Check for consecutive charging: current is charge AND next is charge
        consecutive_charges = is_charging_event[:, :-1] * is_charging_event[:, 1:]  # (B, T-1)
        
        penalty = torch.sum(consecutive_charges) / batch_size
        return penalty
    


class KoVAEWithEVPenalties(nn.Module):
    """
    Extended KoVAE with EV domain-specific penalties
    """
    
    def __init__(self, base_kovae, penalty_config=None, num_features=5, seq_len=10):
        """
        Args:
            base_kovae: Your existing KoVAE model
            penalty_config: Dict with penalty weights
        """
        super(KoVAEWithEVPenalties, self).__init__()
        self.kovae = base_kovae
        self.latent_spec = self.kovae.latent_spec
        self.decoder = self.kovae.decoder
        self.names = self.kovae.names
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Default penalty weights
        self.penalty_config = penalty_config
        
        self.ev_penalties = EVDomainPenalties()
        
        # Track penalty history for monitoring
        self.penalty_history = {k: [] for k in self.penalty_config.keys()}
    
    def forward(self, x, time=None, final_index=None):
        # Call base KoVAE forward
        return self.kovae(x, time, final_index)

    
    def loss(self, x, x_rec, z_dist, z_prior_dist, z_prior_sample, w_kl=None):
        """
        Extended loss with EV domain penalties
        """

        x_flat = x[~torch.isnan(x)]
        x_rec_flat = x_rec[~torch.isnan(x)]

        
        # Get base KoVAE losses
        base_losses = self.kovae.loss(x_flat, x_rec_flat, z_dist, z_prior_dist, z_prior_sample, w_kl)
        total_loss, rec_loss, kl_loss, pred_prior_loss = base_losses
        
        # Compute EV domain penalties
        ev_penalty_total = torch.tensor(0.0, device=x.device)
        penalty_breakdown = {}          
        
        '''# 1. Trip SoC penalty
        if self.penalty_config.get('trip_soc', 0) > 0:
            trip_soc_penalty = self.ev_penalties.compute_trip_soc_penalty(x_rec)
            penalty_breakdown['trip_soc'] = trip_soc_penalty.item()
            ev_penalty_total += self.penalty_config['trip_soc'] * trip_soc_penalty'''
        
        # 2. Consecutive charge penalty
        if self.penalty_config.get('consecutive_charge', 0) > 0:
            consec_charge_penalty = self.ev_penalties.compute_consecutive_charge_penalty(x_rec)
            penalty_breakdown['consecutive_charge'] = consec_charge_penalty.item()
            weighted = self.penalty_config['consecutive_charge'] * consec_charge_penalty
            ev_penalty_total += weighted
        
        # Add EV penalties to total loss
        total_loss_with_penalties = total_loss + ev_penalty_total
        
        # Store penalty history for monitoring
        for k, v in penalty_breakdown.items():
            self.penalty_history[k].append(v)
        
        # Return extended loss tuple
        return (total_loss_with_penalties, rec_loss, kl_loss, pred_prior_loss, 
                ev_penalty_total, penalty_breakdown)
    
    def sample_data(self, n_sample):
        return self.kovae.sample_data(n_sample)
    
    def get_penalty_summary(self):
        """Get summary statistics of penalties during training"""
        import numpy as np
        summary = {}
        for k, v in self.penalty_history.items():
            if len(v) > 0:
                summary[k] = {
                    'mean': np.mean(v),
                    'std': np.std(v),
                    'min': np.min(v),
                    'max': np.max(v),
                    'last': v[-1]
                }
        return summary


