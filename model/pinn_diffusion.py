import torch
import torch.nn as nn
from .pinn import PINNs

class PINNDiffusion(PINNs):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, beta_schedule='linear', num_diffusion_steps=1000):
        super(PINNDiffusion, self).__init__(in_dim, hidden_dim, out_dim, num_layer)
        self.num_diffusion_steps = num_diffusion_steps
        self.beta = self._get_beta_schedule(beta_schedule)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def _get_beta_schedule(self, schedule_type):
        if schedule_type == 'linear':
            scale = 1000 / self.num_diffusion_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.num_diffusion_steps, device=self.device)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")
    
    def q_sample(self, x_0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t]
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise, noise
    
    def p_losses(self, x_0, t):
        """Compute diffusion loss"""
        x_noisy, noise = self.q_sample(x_0, t)
        noise_pred = self.denoise(x_noisy, t)
        return torch.nn.functional.mse_loss(noise, noise_pred)
    
    def denoise(self, x_t, t):
        """Predict noise in the diffusion process"""
        t_embed = t.float() / self.num_diffusion_steps
        t_embed = t_embed.unsqueeze(-1).expand(-1, x_t.size(-1))
        x_input = torch.cat([x_t, t_embed], dim=-1)
        return self.linear(x_input)
    
    def sample(self, shape, device, return_intermediates=False):
        """Sample from the diffusion model"""
        x_t = torch.randn(shape, device=device)
        intermediates = [x_t] if return_intermediates else None
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = self.denoise(x_t, t_tensor)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
                
            x_t = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            
            if return_intermediates:
                intermediates.append(x_t)
                
        return x_t if not return_intermediates else (x_t, intermediates)