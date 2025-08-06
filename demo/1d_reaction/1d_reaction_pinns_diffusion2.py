# 1d_reaction_pinns_diffusion.py
# 用于训练融合PINN和Diffusion的模型

import torch
import torch.nn as nn
import numpy as np
import random
from torch.optim import LBFGS
from tqdm import tqdm
import sys
sys.path.append("../..")
from model.pinn_diffusion import PINNDiffusion
from util import get_data

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda:0'

# 数据准备
res, b_left, b_right, b_upper, b_lower = get_data([0,2*np.pi], [0,1], 101, 101)
res_test, _, _, _, _ = get_data([0,2*np.pi], [0,1], 101, 101)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[:,0:1], res[:,1:2]
x_left, t_left = b_left[:,0:1], b_left[:,1:2]
x_right, t_right = b_right[:,0:1], b_right[:,1:2]
x_upper, t_upper = b_upper[:,0:1], b_upper[:,1:2]
x_lower, t_lower = b_lower[:,0:1], b_lower[:,1:2]

# 模型初始化
model = PINNDiffusion(
    pinn_config=dict(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4),
    diffusion_config=dict(in_dim=3, hidden_dim=256, out_dim=1)
).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

loss_track = []
for i in tqdm(range(500)):
    def closure():
        noise_level = torch.randn_like(x_res) * 0.1

        u_pinn, u_diff = model(x_res, t_res, noise_level)
        u_x = torch.autograd.grad(u_pinn, x_res, grad_outputs=torch.ones_like(u_pinn), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u_pinn, t_res, grad_outputs=torch.ones_like(u_pinn), retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_t - 5 * u_pinn * (1 - u_pinn))**2)
        loss_bc = torch.mean((model.pinn(x_upper, t_upper) - model.pinn(x_lower, t_lower))**2)
        loss_ic = torch.mean((model.pinn(x_left, t_left)[:,0] - torch.exp(-(x_left[:,0] - np.pi)**2 / (2*(np.pi/4)**2)))**2)

        # Diffusion辅助监督loss
        loss_diff = torch.mean((u_pinn - u_diff.detach())**2)

        loss = loss_res + loss_bc + loss_ic + 0.1 * loss_diff
        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss_diff.item()])

        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)

print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}, Loss_Diff: {:4f}'.format(
    loss_track[-1][0], loss_track[-1][1], loss_track[-1][2], loss_track[-1][3]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

torch.save(model.state_dict(), './1dreaction_pinns_diffusion.pt')

# 保存模型和误差评估
model.eval()
x_test = torch.tensor(res_test[:,0:1], dtype=torch.float32, requires_grad=True).to(device)
t_test = torch.tensor(res_test[:,1:2], dtype=torch.float32, requires_grad=True).to(device)
noise_level = torch.zeros_like(x_test)
with torch.no_grad():
    u_pred, _ = model(x_test, t_test, noise_level)
    u_pred = u_pred.cpu().numpy().reshape(101,101)

def h(x):
    return np.exp(-(x-np.pi)**2 / (2*(np.pi/4)**2))

def u_ana(x,t):
    return h(x) * np.exp(5*t) / (h(x) * np.exp(5*t) + 1 - h(x))

u_exact = u_ana(res_test[:,0], res_test[:,1]).reshape(101,101)
rl1 = np.sum(np.abs(u_exact - u_pred)) / np.sum(np.abs(u_exact))
rl2 = np.sqrt(np.sum((u_exact - u_pred)**2) / np.sum(u_exact**2))

print('relative L1 error: {:.6f}'.format(rl1))
print('relative L2 error: {:.6f}'.format(rl2))


# save to json
import json
results = {
    "model_type": "PINNs_Diffusion",
    "Loss_Res": loss_track[-1][0],
    "Loss_BC": loss_track[-1][1],
    "Loss_IC": loss_track[-1][2],
    "Loss_Diff": loss_track[-1][3],
    "Train_Loss": np.sum(loss_track[-1]),
    "relative_L1_error": rl1,
    "relative_L2_error": rl2
}

with open('1d_reaction_pinns_diffusion_result.json', 'w') as f:
    json.dump(results, f, indent=4)

# Visualization
plt.figure(figsize=(4,3))
plt.imshow(pred, extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.savefig('./1dreaction_pinns_diffusion_pred.png')
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(u, extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.savefig('./1dreaction_pinns_diffusion_exact.png')
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(np.abs(pred - u), extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.savefig('./1dreaction_pinns_diffusion_error.png')
plt.show()

# Generate samples using diffusion process
with torch.no_grad():
    samples = model.sample((101, 1), device, return_intermediates=True)[0]
    samples = samples.reshape(101,101).cpu().numpy()

plt.figure(figsize=(4,3))
plt.imshow(samples, extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sampled u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.savefig('./1dreaction_pinns_diffusion_sample.png')
plt.show()