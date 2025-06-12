import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
#import ot as pot
import torch
import torchdyn
from sklearn.cluster import KMeans
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
#from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.utils import *
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_adjacent_moons, generate_moons

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)

# --------- Sample conditional points for one cluster, Eq(14) --------- #

def sample_conditional_pt_1cluster(x0, x1, t, sigma, x0_label, x1_label, cluster):
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))

    x0 = x0[x0_label == cluster]
    x1 = x1[x1_label == cluster]

    # Use labels to condition the sampling
    if x0.shape[0] >= x1.shape[0]:
        t_crop = t[: x1.shape[0]]
        x0 = x0[: x1.shape[0]]
    else:
        x1 = x1[: x0.shape[0]]  # Ensure x0 matches the batch size of x1
        t_crop = t[: x0.shape[0]]

    mu_t = t_crop * x1 + (1 - t_crop) * x0
    epsilon = torch.randn_like(x0)

    return mu_t + sigma * epsilon


# -- Compute the conditional vector field ut(x1|x0) = x1 - x0, Eq(15) -- #

def compute_conditional_vector_field_1cluster(x0, x1, x0_label, x1_label, cluster):
    x0 = x0[x0_label == cluster]
    x1 = x1[x1_label == cluster]

    if x1.shape[0] >= x0.shape[0]:
        x1 = x1[: x0.shape[0]]  # Ensure x1 matches the batch size of x0
    elif x0.shape[0] > x1.shape[0]:
        x0 = x0[: x1.shape[0]]  # Ensure x0 matches the batch size of x1

    return x1 - x0


# ----------------------- Plotting Trajectories ------------------------ #

def my_plot_trajectories(fig, traj, colors):
    """Plot trajectories of some selected samples on a given figure."""
    n = 2000
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c=colors[0])
    ax.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=colors[1])
    ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c=colors[2])
    ax.legend(["Prior sample z(S)", "Flow", "z(0)"])
    ax.set_xticks([])
    ax.set_yticks([])

    # Do not call plt.show() here, so user can manage figure display


# --------------------- Sample 1 quarter of Gaussian ---------------- #

def sample_1gaussian_quarter(n, cluster=0, total_clusters=8):
    """
    Sample from a sector (quarter/octant/etc.) of a Gaussian distribution.
    The first cluster is centered at pi/2, and each subsequent cluster is rotated accordingly.
    """
    gaussian_samples = torch.randn(n, 2)
    theta = torch.atan2(gaussian_samples[:, 1], gaussian_samples[:, 0])  # [-pi, pi]
    theta = (theta + 2 * math.pi) % (2 * math.pi)  # [0, 2pi]
    sector_size = 2 * math.pi / total_clusters
    # Center the first cluster at pi/2
    center = math.pi / 2 + cluster * sector_size
    sector_start = center - sector_size / 2
    sector_end = center + sector_size / 2
    # Handle wrap-around at 2*pi
    mask = ((theta - sector_start) % (2 * math.pi)) < sector_size
    return gaussian_samples[mask]


# ------------------------------ Main core ------------------------------ #

#ot_sampler = OTPlanSampler(method="exact")
sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adjusted learning rate
FM = ConditionalFlowMatcher(sigma=sigma)

start = time.time()

color_list = [
    ["blue", "lightblue", "navy"],
    ["green", "lightgreen", "olive"],
    ["orange", "yellow", "lightcoral"],
    ["brown", "gray", "maroon"],
    ["teal", "lime", "cyan"],
    ["maroon", "coral", "gold"],
    ["darkblue", "darkgreen", "darkred"],
    ["purple", "magenta", "indigo"],
]

quarter = sample_1gaussian_quarter(1000, cluster=0, total_clusters=8)
plt.scatter(
    quarter[:, 0].cpu().numpy(),
    quarter[:, 1].cpu().numpy(),
    c="blue",
    s=10,
    label="Quarter Gaussian",
)
plt.title("Sampled Quarter of Gaussian Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.grid()

for k in range(16000):
    optimizer.zero_grad()

    x0 = sample_1gaussian(batch_size)  # Sample from a single Gaussian
    x1 = sample_8gaussians(batch_size)  # Target distribution is a set of moons

    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(x1)
    #x1 = torch.tensor(kmeans.cluster_centers_).float()  # Use cluster centers as target

    # plt.scatter(x1[:, 0].cpu().numpy(), x1[:, 1].cpu().numpy(), c="red", s=100, label="Target centers")
    # plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), c="blue", s=10, label="Source samples")
    # print(f"Source: {x0.shape}, Target: {x1.shape}")

    cluster_labels = torch.from_numpy(
        kmeans.labels_
    )  # Convert ndarray to torch tensor to use unique()
    cluster_labels = cluster_labels.unique()  # Ensure we have unique cluster labels

    # Draw samples from OT plan
    # x0, x1 = ot_sampler.sample_plan(x0, x1)  # [256 2]

    # print(f"OT plan: {x0.shape}, {x1.shape}")
    # plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), c="blue", s=10, label="OT samples")

    x0_label = kmeans.predict(x0)
    x1_label = kmeans.predict(x1)
     #print("x1 label", x1_label)

    
    x1_label = np.random.randint(0,7, size=(256,))  # Randomly assign labels to x1

    t = torch.rand(x0.shape[0]).type_as(x0)  # Uniformly sample t in [0, 1]

    for cluster in range(0, 7):
        # Sample conditional points for each cluster
        xt = sample_conditional_pt_1cluster(
            x0, x1, t, sigma, x0_label, x1_label, cluster
        )
        ut = compute_conditional_vector_field_1cluster(
            x0, x1, x0_label, x1_label, cluster
        )

        t_crop = t[: xt.shape[0]]
        # print("size xt:", xt.shape, "size t_crop:", t_crop.shape)

        vt = model(torch.cat([xt, t_crop[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)  # MSE loss

        loss.backward()
        optimizer.step()

    if (k + 1) % 2000 == 0:  # Every 5000 iterations
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
            fig = plt.figure(figsize=(6, 6))
            for i in range(len(cluster_labels)):
                # Sample a trajectory from the learned vector field for each cluster
                traj = node.trajectory(
                    sample_1gaussian_quarter(
                        1024, cluster=i, total_clusters=len(cluster_labels)
                    ),
                    t_span=torch.linspace(0, 1, 100),
                )
                # Plot the trajectory
                my_plot_trajectories(fig, traj.cpu().numpy(), colors=color_list[i])
                plt.title(f"Trajectories at iteration {k + 1}")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
            traj = node.trajectory(
                sample_1gaussian(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj.cpu().numpy())
            fig.savefig(f"{savedir}/trajectory_{k}.png")
            # plt.savefig(f"../Figures_Flow_Matching/trajectory_8gaussian_{k}_20000epoch.png")

torch.save(model, f"{savedir}/otcfm_v1.pt")


### plot from the labels and delete in plot_trajectories the last plot
