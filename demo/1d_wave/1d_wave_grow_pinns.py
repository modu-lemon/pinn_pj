import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam
from tqdm import tqdm
import sys
sys.path.append("../..")
from util import *
from model.grow_pinn import GrowPINN1DWave
from model.ntk_utils import NTKPreconditioner, balanced_sum_of_squares
from model.transport import TransportSampler

import argparse

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


def u_ana(x, t):
    # Same analytic solution as baseline
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--transport', type=str, default='off', choices=['off', 'on'])
    parser.add_argument('--transport_every', type=int, default=100)
    parser.add_argument('--num_res', type=int, default=101*101)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'

    # Grid and boundary data
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data([0, 1], [0, 1], 101, 101)
    res_test_np, *_ = get_data([0, 1], [0, 1], 101, 101)

    # Collocation selection (optionally subsample)
    if args.num_res < res_np.shape[0]:
        idx = np.random.choice(res_np.shape[0], args.num_res, replace=False)
        res_np = res_np[idx]

    res = torch.tensor(res_np, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left_np, dtype=torch.float32, requires_grad=True).to(device)
    b_right = torch.tensor(b_right_np, dtype=torch.float32, requires_grad=True).to(device)
    b_upper = torch.tensor(b_upper_np, dtype=torch.float32, requires_grad=True).to(device)
    b_lower = torch.tensor(b_lower_np, dtype=torch.float32, requires_grad=True).to(device)

    x_res, t_res = res[:, 0:1], res[:, 1:2]
    x_left, t_left = b_left[:, 0:1], b_left[:, 1:2]
    x_right, t_right = b_right[:, 0:1], b_right[:, 1:2]
    x_upper, t_upper = b_upper[:, 0:1], b_upper[:, 1:2]
    x_lower, t_lower = b_lower[:, 0:1], b_lower[:, 1:2]

    # Model
    model = GrowPINN1DWave(hidden_dim=512, num_layer=4, use_dual=False).to(device)
    model.apply(init_weights)

    optim = Adam(model.parameters(), lr=args.lr)

    # Preconditioner for two residual components (u_tt and u_xx)
    precond = NTKPreconditioner(num_terms=2, ema_beta=0.9)

    # Optional transport sampler
    sampler = TransportSampler(temperature=1.0, min_uniform_frac=0.1, seed=seed)

    loss_track = []
    pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

    c = 2.0

    for step in tqdm(range(args.steps)):
        optim.zero_grad()

        # PDE residual on collocation points
        pred_res = model(x_res, t_res)
        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

        # Two-term loss: encourage u_tt ≈ c^2 u_xx; we balance terms via preconditioner
        loss_tt = (u_tt.pow(2)).mean()
        loss_xx = ((c * u_xx).pow(2)).mean()
        precond.update([loss_tt.detach(), loss_xx.detach()])
        loss_res_bal, weights = precond.apply([loss_tt, loss_xx], coeffs=[1.0, 1.0])

        # Boundary and initial conditions
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)
        pred_left = model(x_left, t_left)

        # u(x,0) = sin(pi x) + 0.5 sin(3 pi x)
        ic_target = torch.sin(pi * x_left[:, 0]).unsqueeze(-1) + 0.5 * torch.sin(3 * pi * x_left[:, 0]).unsqueeze(-1)
        ui_t = torch.autograd.grad(pred_left, t_left, grad_outputs=torch.ones_like(pred_left), retain_graph=True, create_graph=True)[0]

        loss_ic_1 = (pred_left[:, 0:1] - ic_target).pow(2).mean()
        loss_ic_2 = (ui_t).pow(2).mean()
        loss_ic = loss_ic_1 + loss_ic_2

        loss_bc = pred_upper.pow(2).mean() + pred_lower.pow(2).mean()

        # QoI: energy at t=1 equals energy at t=0 (conservation); match E(t=1) to analytic E(t=0)
        # build test slice at t=1 and t=0
        x_line = torch.linspace(0, 1, 201, device=device).unsqueeze(-1).requires_grad_(True)
        t_one = torch.ones_like(x_line).requires_grad_(True)
        t_zero = torch.zeros_like(x_line).requires_grad_(True)

        u_one = model(x_line, t_one)
        u_zero = model(x_line, t_zero)
        E_one = GrowPINN1DWave.energy_qoi(u_one, x_line, t_one, c=c)

        # analytic energy at t=0
        x_np = x_line.detach().cpu().numpy().reshape(-1)
        # du/dt at t=0 is 0 for this analytic solution; so energy density reduces to c^2 u_x^2
        ux0_np = np.pi * np.cos(np.pi * x_np) + 1.5 * np.pi * np.cos(3 * np.pi * x_np)
        E_exact = (c ** 2) * np.mean(ux0_np ** 2)
        E_exact_t = torch.tensor(E_exact, dtype=torch.float32, device=device)
        loss_qoi = (E_one - E_exact_t).pow(2)

        loss = loss_res_bal + loss_ic + loss_bc + 0.1 * loss_qoi
        loss.backward()
        optim.step()

        loss_track.append([
            float(loss_res_bal.detach().cpu()),
            float(loss_ic.detach().cpu()),
            float(loss_bc.detach().cpu()),
            float(loss_qoi.detach().cpu())
        ])

        # Optional residual-driven resampling
        if args.transport == 'on' and (step + 1) % args.transport_every == 0:
            with torch.no_grad():
                # approximate residual magnitude |u_tt - c^2 u_xx|
                res_mag = (u_tt - (c ** 2) * u_xx).abs().reshape(-1)
                new_points = sampler.resample(res, res_mag, num_samples=res.shape[0])
                res = new_points.detach().clone().to(device)
                res.requires_grad_(True)
                x_res, t_res = res[:, 0:1], res[:, 1:2]

    print('Loss Res(Bal): {:.4f}, Loss_BC: {:.4f}, Loss_IC: {:.4f}, Loss_QoI: {:.4f}'.format(
        loss_track[-1][0], loss_track[-1][2], loss_track[-1][1], loss_track[-1][3]))
    print('Train Loss (sum last): {:.4f}'.format(sum(loss_track[-1])))

    # Save model
    torch.save(model.state_dict(), './1dwave_grow_pinns.pt')

    # Evaluate on grid
    res_test = torch.tensor(res_test_np, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1]
        pred = pred.cpu().detach().numpy()

    pred = pred.reshape(101, 101)

    u = u_ana(res_test_np[:, 0], res_test_np[:, 1]).reshape(101, 101)

    rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

    print('relative L1 error: {:4f}'.format(rl1))
    print('relative L2 error: {:4f}'.format(rl2))

    # save to json
    import json
    results = {
        "model_type": "GROW-PINNs",
        "Loss_Res_bal": loss_track[-1][0],
        "Loss_BC": loss_track[-1][2],
        "Loss_IC": loss_track[-1][1],
        "Loss_QoI": loss_track[-1][3],
        "Train_Loss_sum_last": float(sum(loss_track[-1])),
        "relative_L1_error": rl1,
        "relative_L2_error": rl2,
        "weights_last": [float(w.item()) for w in precond.get_weights().detach().cpu().reshape(-1)]
    }

    with open('1d_wave_grow_pinns_result.json', 'w') as f:
        json.dump(results, f, indent=4)

    if HAS_PLT:
        # Visualize
        plt.figure(figsize=(4, 3))
        plt.imshow(pred, extent=[0, 1, 1, 0])
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Predicted u(x,t) - GROW-PINN')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./1dwave_grow_pinns_pred.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(u, extent=[0, 1, 1, 0])
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./1dwave_exact.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(np.abs(pred - u), extent=[0, 1, 1, 0])
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Absolute Error')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./1dwave_grow_pinns_error.png')


if __name__ == "__main__":
    main()