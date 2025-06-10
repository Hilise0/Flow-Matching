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
from sklearn.cluster import KMeans

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)


def sample_conditional_pt_1cluster(x0, x1, t, sigma, x0_label, x1_label, cluster):
    mu_t = 0
    epsilon = 0
    if x0_label.any() == cluster and x1_label.any() == cluster:
        x0 = x0[x0_label == cluster]
        x1 = x1[x1_label == cluster]

        # Use labels to condition the sampling
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon


def compute_conditional_vector_field_1cluster(x0, x1, x0_label, x1_label, cluster):
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
    if x0_label == cluster and x1_label == cluster:
        x0 = x0[x0_label == cluster]
        x1 = x1[x1_label == cluster]
    return x1 - x0


ot_sampler = OTPlanSampler(method="exact")
sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

start = time.time()

for k in range(10000):
    optimizer.zero_grad()

    x0 = sample_1gaussian(batch_size)  # Sample from a single Gaussian
    x11 = sample_8gaussians(batch_size)  # Target distribution is a set of moons

    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(x11)
    x1 = torch.tensor(kmeans.cluster_centers_).float()  # Use cluster centers as target
    # plt.scatter(x1[:, 0].cpu().numpy(), x1[:, 1].cpu().numpy(), c="red", s=100, label="Target centers")
    # plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), c="blue", s=10, label="Source samples")
    print(f"Source: {x0.shape}, Target: {x1.shape}")
    cluster_labels = torch.from_numpy(
        kmeans.labels_
    )  # Convert ndarray to torch tensor to use unique()
    cluster_labels = cluster_labels.unique()  # Ensure we have unique cluster labels

    # Draw samples from OT plan
    x0, x1 = ot_sampler.sample_plan(x0, x1)  # [256 2]
    print(f"OT plan: {x0.shape}, {x1.shape}")
    # plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), c="blue", s=10, label="OT samples")

    x0_label = kmeans.predict(x0)
    x1_label = kmeans.predict(x1)
    t = torch.rand(x0.shape[0]).type_as(x0)  # Uniformly sample t in [0, 1]

    for cluster in range(1, 7):
        # Sample conditional points for each cluster
        xt = sample_conditional_pt_1cluster(x0, x1, t, sigma, x0_label, x1_label, cluster)
        ut = compute_conditional_vector_field_1cluster(x0, x1, x0_label, x1_label, cluster)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)  # MSE loss

        loss.backward()
        optimizer.step()

    if (k + 1) % 1000 == 0:  # Every 5000 iterations
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
