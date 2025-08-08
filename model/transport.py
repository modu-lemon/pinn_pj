import torch
import numpy as np
from typing import Tuple


def residual_importance_sampling(points: torch.Tensor,
                                 residual_values: torch.Tensor,
                                 num_samples: int,
                                 temperature: float = 1.0,
                                 min_uniform_frac: float = 0.1,
                                 rng: np.random.RandomState = None) -> torch.Tensor:
    """
    Sample a subset of collocation points proportional to residual magnitude.
    points: (N, D) tensor on device (x, t) or (x, y, t)
    residual_values: (N,) or (N,1) tensor of nonnegative magnitudes
    returns: (num_samples, D) tensor picked from points (no gradients through sampling)
    """
    if rng is None:
        rng = np.random.RandomState(0)

    with torch.no_grad():
        N = points.shape[0]
        res = residual_values.detach().reshape(N).abs().cpu().numpy()
        res = res / (res.mean() + 1e-12)
        logits = np.log(res + 1e-12) / max(temperature, 1e-3)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        # Mix with uniform to avoid collapse
        uniform = np.ones_like(probs) / len(probs)
        probs = (1.0 - min_uniform_frac) * probs + min_uniform_frac * uniform
        probs = probs / probs.sum()

        idx = rng.choice(len(probs), size=min(num_samples, N), replace=False, p=probs)
        sampled = points[idx]
    return sampled


class TransportSampler:
    def __init__(self, temperature: float = 1.0, min_uniform_frac: float = 0.1, seed: int = 0):
        self.temperature = temperature
        self.min_uniform_frac = min_uniform_frac
        self.rng = np.random.RandomState(seed)

    def resample(self, points: torch.Tensor, residual_values: torch.Tensor, num_samples: int) -> torch.Tensor:
        return residual_importance_sampling(points, residual_values, num_samples,
                                            temperature=self.temperature,
                                            min_uniform_frac=self.min_uniform_frac,
                                            rng=self.rng)