import torch
import torch.nn as nn
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layer: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers += [nn.Linear(in_dim, hidden_dim), activation]
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GrowPINN1DWave(nn.Module):
    """
    Minimal wrapper for 1D wave equation u_tt - c^2 u_xx = 0.
    Provides:
      - primal network u(x,t)
      - optional dual network z(x,t) (not used by default, kept for extension)
      - utility to compute energy-based QoI at a given time slice
    """

    def __init__(self, hidden_dim: int = 512, num_layer: int = 4, use_dual: bool = False):
        super().__init__()
        self.primal = MLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=num_layer)
        self.use_dual = use_dual
        if use_dual:
            self.dual = MLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=num_layer)
        else:
            self.dual = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        src = torch.cat((x, t), dim=-1)
        return self.primal(src)

    @staticmethod
    def energy_qoi(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, c: float = 2.0) -> torch.Tensor:
        """
        Compute energy integral E = ∫ (u_t^2 + c^2 u_x^2) dx over the domain x in [0,1].
        Uses mean over discrete points and rescales by domain length.
        """
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        density = u_t.pow(2) + (c ** 2) * u_x.pow(2)
        # approximate integral via average times length (1.0)
        return density.mean()