import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.utils import *
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_adjacent_moons, generate_moons

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon


def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0


ot_sampler = OTPlanSampler(method="exact")
sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Learning rate for the optimizer
FM = ConditionalFlowMatcher(sigma=sigma)

start = time.time()

for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_1gaussian(batch_size)  # Sample from a single Gaussian
    x1 = sample_8gaussians(batch_size)  # Target distribution is a set of moons

    # Draw samples from OT plan
    x0, x1 = ot_sampler.sample_plan(x0, x1)

    t = torch.rand(x0.shape[0]).type_as(x0)  # Uniformly sample t in [0, 1]
    xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
    ut = compute_conditional_vector_field(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)  # MSE loss

    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:  # Every 5000 iterations
        end = time.time()
        print(f"{k + 1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end

        # Solve the ODE with the learned vector field
        node = NeuralODE(
            torch_wrapper(model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        with torch.no_grad():
            # Sample a trajectory from the learned vector field
            traj = node.trajectory(
                sample_1gaussian(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            # Plot the trajectory
            plot_trajectories(traj.cpu().numpy())
torch.save(model, f"{savedir}/otm_v1.pt")
