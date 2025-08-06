# import torch
# import torch.nn as nn
# from .pinn import PINNs

# class PINNDiffusion(PINNs):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_layer, beta_schedule='linear', num_diffusion_steps=1000):
#         super(PINNDiffusion, self).__init__(in_dim, hidden_dim, out_dim, num_layer)
#         self.device = next(self.parameters()).device
#         self.num_diffusion_steps = num_diffusion_steps
#         self.beta = self._get_beta_schedule(beta_schedule).to(self.device)
#         self.alpha = (1 - self.beta).to(self.device)
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)
    
#     def _get_beta_schedule(self, schedule_type):
#         if schedule_type == 'linear':
#             scale = 1000 / self.num_diffusion_steps
#             beta_start = scale * 0.0001
#             beta_end = scale * 0.02
#             return torch.linspace(beta_start, beta_end, self.num_diffusion_steps, device=self.device)
#         else:
#             raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")
    
#     def forward(self, x, t):
#         """Override parent class forward to handle input"""
#         if isinstance(t, torch.Tensor) and t.dim() == 1:
#             t = t.view(-1, 1)
#         if isinstance(x, torch.Tensor) and x.dim() == 1:
#             x = x.view(-1, 1)
#         src = torch.cat((x, t), dim=-1)
#         return self.linear(src)
    
#     def q_sample(self, x_0, t):
#         """Forward diffusion process"""
#         x_0 = x_0.to(self.device)
#         t = t.to(self.device)
#         noise = torch.randn_like(x_0, device=self.device)
#         alpha_bar_t = self.alpha_bar[t].view(-1, 1)
#         return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise, noise
    
#     def p_losses(self, x_0, t):
#         """Compute diffusion loss"""
#         x_0 = x_0.to(self.device)
#         t = t.to(self.device)
#         if x_0.dim() == 1:
#             x_0 = x_0.unsqueeze(-1)
#         x_noisy, noise = self.q_sample(x_0, t)
#         noise_pred = self.denoise(x_noisy, t)
#         return torch.nn.functional.mse_loss(noise, noise_pred)
    
#     def denoise(self, x_t, t):
#         """Predict noise in the diffusion process"""
#         x_t = x_t.to(self.device)
#         t = t.to(self.device)
#         if x_t.dim() == 1:
#             x_t = x_t.unsqueeze(-1)
#         if t.dim() == 1:
#             t = t.unsqueeze(-1)
#         t_embed = t.float() / self.num_diffusion_steps
#         # 保证拼接的两个张量都在同一设备
#         x_input = torch.cat([x_t, t_embed], dim=-1)
#         return self.linear(x_input)
    
#     def sample(self, shape, device=None, return_intermediates=False):
#         """Sample from the diffusion model"""
#         device = device or self.device
#         x_t = torch.randn(shape, device=device)
#         intermediates = [x_t] if return_intermediates else None
        
#         for t in reversed(range(self.num_diffusion_steps)):
#             t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
#             noise_pred = self.denoise(x_t, t_tensor)
            
#             alpha_t = self.alpha[t].to(device)
#             alpha_bar_t = self.alpha_bar[t].to(device)
#             beta_t = self.beta[t].to(device)
            
#             if t > 0:
#                 noise = torch.randn_like(x_t, device=device)
#             else:
#                 noise = torch.zeros_like(x_t, device=device)
            
#             alpha_t = alpha_t.view(-1, 1)
#             alpha_bar_t = alpha_bar_t.view(-1, 1)
#             beta_t = beta_t.view(-1, 1)
            
#             x_t = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            
#             if return_intermediates:
#                 intermediates.append(x_t)
                
#         return x_t if not return_intermediates else (x_t, intermediates)



# pinn_diffusion.py
# 融合PINN和Diffusion的网络模块定义

import torch
import torch.nn as nn

# PINN模块（用于PDE残差计算）
class PINNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNBlock, self).__init__()
        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        input_ = torch.cat([x, t], dim=1)
        return self.net(input_)

# 简化版Diffusion生成器（UNet或简单MLP）
class DiffusionGenerator(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=256, out_dim=1):
        super(DiffusionGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, t, noise_level):
        input_ = torch.cat([x, t, noise_level], dim=1)  # 输入拼接x,t,noise_level
        return self.net(input_)

# 整合的PINN + Diffusion模型
class PINNDiffusion(nn.Module):
    def __init__(self, pinn_config, diffusion_config):
        super(PINNDiffusion, self).__init__()
        self.pinn = PINNBlock(**pinn_config)
        self.diffusion = DiffusionGenerator(**diffusion_config)

    def forward(self, x, t, noise_level):
        pinn_out = self.pinn(x, t)
        diffusion_out = self.diffusion(x, t, noise_level)
        return pinn_out, diffusion_out
