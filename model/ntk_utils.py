import torch
from typing import List, Optional, Tuple


class NTKPreconditioner:
    """
    Lightweight residual preconditioner using EMA-balancing.
    It approximates NTK-based spectrum shaping by reweighting per-term losses
    to equalize their magnitudes, improving optimization conditioning.

    This avoids computing full NTK; instead, it tracks an EMA of each term's
    scalar loss and uses inverse sqrt of the EMA as weights.
    """

    def __init__(self, num_terms: int, ema_beta: float = 0.9, eps: float = 1e-8):
        self.ema_beta = ema_beta
        self.eps = eps
        self.registered = False
        self.num_terms = num_terms
        self.ema_values = [None for _ in range(num_terms)]

    @torch.no_grad()
    def update(self, term_losses: List[torch.Tensor]) -> None:
        assert len(term_losses) == self.num_terms, "term_losses length must match num_terms"
        for i, loss_i in enumerate(term_losses):
            # Convert to scalar tensor
            scalar = loss_i.detach()
            if scalar.ndim > 0:
                scalar = scalar.mean()
            if self.ema_values[i] is None:
                self.ema_values[i] = scalar
            else:
                self.ema_values[i] = self.ema_beta * self.ema_values[i] + (1.0 - self.ema_beta) * scalar

    def get_weights(self) -> torch.Tensor:
        # w_i = 1/sqrt(EMA_i + eps), then normalized to have mean 1
        device = None
        vals = []
        for v in self.ema_values:
            if v is None:
                vals.append(torch.tensor(1.0))
            else:
                vals.append(v)
            if device is None and v is not None:
                device = v.device
        if device is None:
            device = torch.device('cpu')
        ema_tensor = torch.stack([vv.to(device) for vv in vals])
        raw = torch.rsqrt(ema_tensor + self.eps)
        return raw / (raw.mean() + self.eps)

    def apply(self, term_losses: List[torch.Tensor], coeffs: Optional[List[float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns weighted sum of term losses and the weights used.
        coeffs can be used to encode PDE coefficients per term.
        """
        if coeffs is None:
            coeffs = [1.0 for _ in term_losses]
        assert len(term_losses) == self.num_terms
        assert len(coeffs) == self.num_terms

        weights = self.get_weights().to(term_losses[0].device)
        total = 0.0
        for i, loss_i in enumerate(term_losses):
            total = total + weights[i] * float(coeffs[i]) * loss_i
        return total, weights


def balanced_sum_of_squares(terms: List[torch.Tensor], coeffs: Optional[List[float]] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Fallback: balance by normalizing each term by its RMS, then sum squares.
    This is stateless and can be used without the EMA preconditioner.
    """
    if coeffs is None:
        coeffs = [1.0 for _ in terms]
    assert len(coeffs) == len(terms)

    total = 0.0
    for t, c in zip(terms, coeffs):
        rms = torch.sqrt(t.pow(2).mean() + eps)
        total = total + (c * t / (rms + eps)).pow(2).mean()
    return total