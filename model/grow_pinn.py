import torch
import torch.nn as nn
from typing import Tuple, Optional


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


class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, num_frequencies: int = 6, max_freq: float = 16.0):
        super().__init__()
        # Frequencies are powers of two up to max_freq
        self.freqs = nn.Parameter(torch.linspace(1.0, max_freq, num_frequencies), requires_grad=False)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        parts = [x]
        for f in self.freqs:
            parts.append(torch.sin(2 * torch.pi * f * x))
            parts.append(torch.cos(2 * torch.pi * f * x))
        return torch.cat(parts, dim=-1)

    def out_dim(self) -> int:
        return self.in_dim + 2 * self.in_dim * len(self.freqs)


class FourierFeatureMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 256, out_dim: int = 1, num_layer: int = 6,
                 num_frequencies: int = 6, max_freq: float = 16.0, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.encoder = FourierFeatureEncoder(in_dim=in_dim, num_frequencies=num_frequencies, max_freq=max_freq)
        enc_dim = self.encoder.out_dim()
        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers += [nn.Linear(enc_dim, hidden_dim), activation]
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.net(z)


class GrowPINN1DWave(nn.Module):
    """
    1D wave equation u_tt - c^2 u_xx = 0.
    Provides:
      - primal network u(x,t)
      - optional dual network z(x,t) (not used by default, kept for extension)
      - utility to compute energy-based QoI at a given time slice
    """

    def __init__(self, hidden_dim: int = 512, num_layer: int = 4, use_dual: bool = False,
                 use_fourier: bool = True, num_frequencies: int = 6, max_freq: float = 16.0):
        super().__init__()
        if use_fourier:
            self.primal = FourierFeatureMLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=max(4, num_layer+2),
                                            num_frequencies=num_frequencies, max_freq=max_freq, activation=nn.SiLU())
        else:
            self.primal = MLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=num_layer)
        self.use_dual = use_dual
        if use_dual:
            if use_fourier:
                self.dual = FourierFeatureMLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=max(4, num_layer+2),
                                              num_frequencies=num_frequencies, max_freq=max_freq, activation=nn.SiLU())
            else:
                self.dual = MLP(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=num_layer)
        else:
            self.dual = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        src = torch.cat((x, t), dim=-1)
        return self.primal(src)

    @staticmethod
    def energy_qoi(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, c: float = 2.0) -> torch.Tensor:
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        density = u_t.pow(2) + (c ** 2) * u_x.pow(2)
        return density.mean()