# GROW-PINN: Goal-oriented Residual-Optimal Weighted PINN (minimal demo)

This repository extends the baseline PINN demos with a minimal, runnable implementation of the proposed GROW-PINN ideas:

- NTK-style residual preconditioning via EMA balancing (`model/ntk_utils.py`)
- Optional residual-importance transport sampling (`model/transport.py`)
- Goal-oriented QoI term (energy conservation) for 1D Wave in `demo/1d_wave/1d_wave_grow_pinns.py`

The demos mirror the style of existing scripts in `demo/*` and produce JSON/PNG outputs.

## Environment
- Python 3.8+
- PyTorch 1.10+
- NumPy, Matplotlib, SciPy (for Navier–Stokes demo)

## Quickstart

### 1D Wave (GROW-PINN)
Runs Adam for 2000 steps, with EMA preconditioning and optional transport resampling.

```bash
python demo/1d_wave/1d_wave_grow_pinns.py --steps 2000 --lr 1e-3 --device cuda:0 --transport off
```

Outputs:
- `1dwave_grow_pinns.pt` model weights
- `1d_wave_grow_pinns_result.json` metrics
- `1dwave_grow_pinns_pred.png`, `1dwave_grow_pinns_error.png`

### 1D Convection (toy) (GROW-PINN)
```bash
python demo/convection/convection_grow_pinns.py
```

### 1D Reaction-Diffusion (toy) (GROW-PINN)
```bash
python demo/1d_reaction/1d_reaction_grow_pinns.py
```

### 2D Navier–Stokes (cylinder wake) (GROW-PINN)
Requires `demo/navier_stokes/cylinder_nektar_wake.mat` (as in the baseline demo).
```bash
python demo/navier_stokes/naiver_stoke_grow_pinns.py
```

## Configs
Example YAML for 1D Wave at `config/1d_wave_grow.yaml`. You can load and map these to CLI flags in your own launcher if needed.

## Notes
- The NTK preconditioning here is a lightweight surrogate (EMA-based balancing) to improve optimization conditioning without computing full NTK.
- Transport sampling is optional and resamples collocation points by residual magnitudes.
- The 1D Wave demo includes a goal-oriented QoI term (energy conservation) to reflect the proposed framework.