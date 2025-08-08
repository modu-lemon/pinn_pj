import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam
from tqdm import tqdm
import sys
sys.path.append("../..")
from util import *
from model.grow_pinn import GrowPINN1DWave  # reuse MLP structure for a 1D PDE baseline
from model.ntk_utils import NTKPreconditioner

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# Minimal convection-like PDE example on [0,1]x[0,1]: u_t + a u_x = 0 with periodic BC
# Analytic solution for u(x,t) given u(x,0) = sin(2pi x): u(x,t) = sin(2pi(x - a t))

a = 1.0


def u_ana(x, t):
    return np.sin(2 * np.pi * (x - a * t))


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Grid
res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data([0, 1], [0, 1], 128, 128)
res = torch.tensor(res_np, dtype=torch.float32, requires_grad=True).to(device)
res_test_np, *_ = get_data([0, 1], [0, 1], 128, 128)

x_res, t_res = res[:, 0:1], res[:, 1:2]

# Model (reuse GrowPINN1DWave's MLP for convenience)
model = GrowPINN1DWave(hidden_dim=256, num_layer=4, use_dual=False).to(device)
model.apply(init_weights)
optim = Adam(model.parameters(), lr=1e-3)
precond = NTKPreconditioner(num_terms=2)

loss_track = []

for step in tqdm(range(1500)):
    optim.zero_grad()
    pred = model(x_res, t_res)
    ones = torch.ones_like(pred)
    u_x = torch.autograd.grad(pred, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(pred, t_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]

    # Residual: u_t + a u_x = 0, balance the two components
    loss_t = (u_t.pow(2)).mean()
    loss_x = ((a * u_x).pow(2)).mean()
    precond.update([loss_t.detach(), loss_x.detach()])
    loss_res, _ = precond.apply([loss_t, loss_x])

    # Periodic BC along x: u(0,t) = u(1,t)
    x0 = torch.zeros_like(t_res)
    x1 = torch.ones_like(t_res)
    u0 = model(x0, t_res)
    u1 = model(x1, t_res)
    loss_bc = (u0 - u1).pow(2).mean()

    # IC: u(x,0) = sin(2pi x)
    x_line = torch.linspace(0, 1, 256, device=device).unsqueeze(-1).requires_grad_(True)
    t0 = torch.zeros_like(x_line)
    u_ic = model(x_line, t0)
    ic_target = torch.sin(2 * np.pi * x_line)
    loss_ic = (u_ic - ic_target).pow(2).mean()

    loss = loss_res + loss_bc + loss_ic
    loss.backward()
    optim.step()

    loss_track.append([float(loss_res.detach().cpu()), float(loss_bc.detach().cpu()), float(loss_ic.detach().cpu())])

# Evaluate
res_test = torch.tensor(res_test_np, dtype=torch.float32, requires_grad=False).to(device)
x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]
with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1].cpu().numpy()

u = u_ana(res_test_np[:, 0], res_test_np[:, 1])
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))
print('Convection relative L2 error: {:.4f}'.format(rl2))

# Save figure (optional)
if HAS_PLT:
    pred_img = pred.reshape(128, 128)
    plt.figure(figsize=(4, 3))
    plt.imshow(pred_img, extent=[0, 1, 1, 0])
    plt.title('Convection u(x,t) - GROW-PINN')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./convection_grow_pinns_pred.png')