import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import scipy.io
import sys
sys.path.append("../..")
from util import *
from model.pinn_diffusion import PINNDiffusion

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = 'cuda:0'

data = scipy.io.loadmat('./cylinder_nektar_wake.mat')

U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

idx = np.random.choice(N*T,2500, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)

class PINNsDiffusionNS(PINNDiffusion):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, num_diffusion_steps=100):
        super(PINNsDiffusionNS, self).__init__(in_dim, hidden_dim, out_dim, num_layer, num_diffusion_steps=num_diffusion_steps)

    def forward(self, x, y, t):
        src = torch.cat((x,y,t), dim=-1)
        return self.linear(src)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Train PINNs with Diffusion
model = PINNsDiffusionNS(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4, num_diffusion_steps=100).to(device)
model.apply(init_weights)
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(get_n_params(model))

loss_track = []

for i in tqdm(range(1000)):
    def closure():
        # Sample timestep for diffusion
        t_diff = torch.randint(0, model.num_diffusion_steps, (x_train.shape[0],), device=device)
        
        # Forward predictions
        psi_and_p = model(x_train, y_train, t_train)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]

        u = torch.autograd.grad(psi, y_train, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = - torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

        u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_train, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x_train, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y_train, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        f_u = u_t + (u*u_x + v*u_y) + p_x - 0.01*(u_xx + u_yy) 
        f_v = v_t + (u*v_x + v*v_y) + p_y - 0.01*(v_xx + v_yy)

        # Data loss
        loss_data = torch.mean((u - u_train)**2) + torch.mean((v - v_train)**2)
        # PDE loss
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)
        # Diffusion loss
        loss_diff = model.p_losses(psi_and_p, t_diff)

        loss = loss_data + loss_pde + 0.1 * loss_diff

        loss_track.append([loss_data.item(), loss_pde.item(), loss_diff.item()])

        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)

print('Loss Data: {:4f}, Loss PDE: {:4f}, Loss Diff: {:4f}'.format(
    loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

torch.save(model.state_dict(), './ns_pinns_diffusion.pt')

# save to json
import json
results = {
    "model_type": "PINNs_Diffusion",
    "Loss_Data": loss_track[-1][0],
    "Loss_PDE": loss_track[-1][1],
    "Loss_Diff": loss_track[-1][2],
    "Train_Loss": np.sum(loss_track[-1])
}

with open('ns_pinns_diffusion_result.json', 'w') as f:
    json.dump(results, f, indent=4)

# Visualization
idx_t = [0,50,100,150,200]

for i in idx_t:
    plt.figure(figsize=(6,5))
    x_star = X_star[:,0]
    y_star = X_star[:,1]
    u_pred = U_star[:,0,i]
    
    plt.scatter(x_star, y_star, c=u_pred)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('u(t,x,y) at t = {:.2f}'.format(t_star[i,0]))
    plt.tight_layout()
    plt.savefig(f'./ns_pinns_diffusion_u_{i}.png')
    plt.show()

# Generate samples using diffusion process
with torch.no_grad():
    samples = model.sample((N, 2), device, return_intermediates=True)[0]
    samples = samples.cpu().numpy()

plt.figure(figsize=(6,5))
plt.scatter(x_star, y_star, c=samples[:,0])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sampled u(t,x,y)')
plt.tight_layout()
plt.savefig('./ns_pinns_diffusion_sample.png')
plt.show()