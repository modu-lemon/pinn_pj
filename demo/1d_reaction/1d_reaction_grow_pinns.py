import numpy as np
import torch
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False
from torch.optim import Adam
from tqdm import tqdm
import sys
sys.path.append("../..")
from util import *
from model.grow_pinn import GrowPINN1DWave  # reuse MLP
from model.ntk_utils import NTKPreconditioner

# Simple reaction-diffusion: u_t - D u_xx + k u = 0 on [0,1]x[0,1]
D = 0.1
k = 1.0

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data([0, 1], [0, 1], 128, 128)
res = torch.tensor(res_np, dtype=torch.float32, requires_grad=True).to(device)
res_test_np, *_ = get_data([0, 1], [0, 1], 128, 128)

x_res, t_res = res[:, 0:1], res[:, 1:2]

model = GrowPINN1DWave(hidden_dim=256, num_layer=4, use_dual=False).to(device)
for m in model.modules():
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

optim = Adam(model.parameters(), lr=1e-3)
precond = NTKPreconditioner(num_terms=3)

for step in tqdm(range(1500)):
    optim.zero_grad()
    pred = model(x_res, t_res)
    ones = torch.ones_like(pred)
    u_x = torch.autograd.grad(pred, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(pred, t_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]

    # Terms: u_t, D u_xx, k u
    loss_t = (u_t.pow(2)).mean()
    loss_xx = ((D * u_xx).pow(2)).mean()
    loss_rxn = ((k * pred).pow(2)).mean()

    precond.update([loss_t.detach(), loss_xx.detach(), loss_rxn.detach()])
    loss_res, _ = precond.apply([loss_t, loss_xx, loss_rxn])

    # IC: u(x,0) = sin(pi x)
    x_line = torch.linspace(0, 1, 256, device=device).unsqueeze(-1).requires_grad_(True)
    t0 = torch.zeros_like(x_line)
    u_ic = model(x_line, t0)
    ic_target = torch.sin(np.pi * x_line)
    loss_ic = (u_ic - ic_target).pow(2).mean()

    # Homogeneous Dirichlet BC u(0,t)=u(1,t)=0
    t_line = torch.linspace(0, 1, 256, device=device).unsqueeze(-1)
    x0 = torch.zeros_like(t_line)
    x1 = torch.ones_like(t_line)
    loss_bc = model(x0, t_line).pow(2).mean() + model(x1, t_line).pow(2).mean()

    loss = loss_res + loss_ic + loss_bc
    loss.backward()
    optim.step()

# Evaluate against a surrogate numerical solution (optional): here we compute only smoothness proxy
res_test = torch.tensor(res_test_np, dtype=torch.float32, requires_grad=False).to(device)
x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]
with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1].cpu().numpy()

if HAS_PLT:
    pred_img = pred.reshape(128, 128)
    plt.figure(figsize=(4, 3))
    plt.imshow(pred_img, extent=[0, 1, 1, 0])
    plt.title('1D Reaction-Diffusion - GROW-PINN')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./1d_reaction_grow_pinns_pred.png')