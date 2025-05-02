from copy import deepcopy
from pathlib import Path

import gudhi as gd  # type: ignore
import numpy as np
import pandas as pd
from gudhi.representations import Entropy
from gudhi.representations.vector_methods import Landscape
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import rcParams

from fmtda import Metric, SimplexTreeBuilder, utils

rcParams["font.family"] = "serif"
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.figsize"] = (7, 7)
rcParams["figure.dpi"] = 300
rcParams["axes.titlesize"] = 15


# datapath
data_path = Path(__file__).parent / "Clinical_fm_66_.xlsx"
# set random seet for reproducability
np.random.seed(32)
# read patient data
patientData = pd.read_excel(data_path, sheet_name="data_66")
max_dim = 2
entropies = {}
st_rips_per_metric = {}

# set weights for metrics.
constant_arrays = [
    np.concatenate((np.random.random(1), np.ones(2))),
    np.concatenate((np.random.random(1), np.ones(2))),
    np.concatenate((np.random.random(1), np.ones(2))),
    np.random.random(1),
    np.random.random(2),
    np.random.random(2),
    np.random.random(1),
    np.random.random(2),
]
w = np.random.random(8)
copy_constant_arrays = deepcopy(constant_arrays)
constant_arrays.append((w, copy_constant_arrays))  # type: ignore
sizes = []

for i, c in enumerate(constant_arrays):
    # instantiate metric
    metric = Metric(i + 1, c)
    print(f"Metric: {i+1}")

    feature_fun = eval(f"utils.feature_{i + 1}")

    x_vals, feature_set, transforms = feature_fun(patientData)

    x_df = pd.DataFrame(data=x_vals)
    x_df.dropna(axis=0, inplace=True)

    data = x_df.values
    print(f"Data shape: {data.shape}")
    if metric.type != 9:
        sizes.append(data.shape[1])
        distMat = metric.dist_matrix(data)
    else:
        distMat = metric.dist_matrix(data, sizes=sizes)

    print(f"Dist matrix shape: {distMat.shape}")

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist) / 8

    print(f"minDist: {minDist}")
    print(f"maxDist: {maxDist}")
    print(f"interval: {interval}")

    st_rips = gd.RipsComplex(
        distance_matrix=distMat, max_edge_length=np.inf
    ).create_simplex_tree(max_dimension=max_dim)

    diagram_rips = st_rips.persistence(
        homology_coeff_field=2, persistence_dim_max=True, min_persistence=1.45
    )

    st_rips_per_metric[i] = diagram_rips
    
    # Persistence Diagram Analysis via Persistence Entropy + Landscapes
    entropies[i] = {}
    for d in range(max_dim):
        try:
            birth_death_pairs = np.array(
                [pair[1] for pair in diagram_rips if pair[0] == d]
            )
            if birth_death_pairs.ndim != 2 or birth_death_pairs.shape[1] != 2:
                raise ValueError(f"No valid birth-death pairs in H{d}")

            # Filter out pairs that die immediately or are infinite
            lifetimes = birth_death_pairs[:, 1] - birth_death_pairs[:, 0]
            finite_mask = np.isfinite(lifetimes)
            nonzero_mask = lifetimes > 1e-6
            valid_pairs = birth_death_pairs[finite_mask & nonzero_mask]

            if len(valid_pairs) == 0:
                raise ValueError(f"All lifetimes filtered out in H{d}")

            entropy = Entropy(mode="scalar")
            persistence_entropy = entropy(valid_pairs)
            entropies[i][f"H{d}"] = persistence_entropy
            print(
                f"Persistence Entropy for Metric {i+1} on H{d}: {persistence_entropy}"
            )

            # Plotting Persistence Landscapes for H0
            if d == 0:
                landscape = Landscape(num_landscapes=5, resolution=100)
                landscape.fit([valid_pairs])
                pl_vector = landscape.transform([valid_pairs])[0]
                nl, res = landscape.num_landscapes, landscape.resolution
                pl_matrix = pl_vector.reshape(nl, res)
                grid = landscape.grid_

                plt.figure(figsize=(6, 4))
                for k in range(nl):
                    plt.plot(grid, pl_matrix[k], label=f"lambda$_{{{k+1}}}$")
                plt.title(
                    rf"Persistence Landscape (H$_{{0}}$) for Metric {i+1}"
                )
                plt.xlabel("Filtration value")
                plt.ylabel("Landscape value")
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(f"landscape_metric_{i+1}_H0.png")
                plt.close()

        except Exception as e:
            print(f"Error computing H{d} for Metric {i+1}: {e}")
            entropies[i][f"H{d}"] = None

    gd.plot_persistence_diagram(diagram_rips, alpha=0.3)
    _ = plt.title(f"Persistence Diagram for Metric {i+1}")
    plt.savefig(f"persistence_diagram_{i+1}.png")

# Persistence Entropy plot for metrics 1,2,3,5,8
metrics = [1, 2, 3, 5, 8]
idxs    = [m - 1 for m in metrics]
labels  = [f"Metric {m}" for m in metrics]

def _scalar_or_nan(v):
    if v is None:
        return 0.0
    if isinstance(v, (list, tuple, np.ndarray)):
        return float(v[0])
    return float(v)

h0 = [_scalar_or_nan(entropies.get(i, {}).get("H0")) for i in idxs]
h1 = [_scalar_or_nan(entropies.get(i, {}).get("H1")) for i in idxs]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
ax0.bar(labels, h0, color="skyblue")
ax0.set_title(r"Persistence Entropy $H_0$")
ax0.set_ylabel("Entropy")
ax0.tick_params(axis="x", rotation=45)

ax1.bar(labels, h1, color="salmon")
ax1.set_title(r"Persistence Entropy $H_1$")
ax1.tick_params(axis="x", rotation=45)

plt.suptitle("Persistence Entropy for Metrics 1, 2, 3, 5, 8", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("persistence_entropy_subplots.png")
plt.close()

# Persistence landscape subplots for metrics 1,2,3,5,8
fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharey=True)
axes = axes.flatten()

for ax, m in zip(axes, metrics):
    idx = m - 1
    diagram = st_rips_per_metric.get(idx)
    if diagram is None:
        ax.axis("off")
        continue

    bd = np.array([p[1] for p in diagram if p[0] == 0])
    lifetimes = bd[:, 1] - bd[:, 0]
    valid = bd[np.isfinite(lifetimes) & (lifetimes > 1e-6)]
    if valid.size == 0:
        ax.axis("off")
        continue

    L = Landscape(num_landscapes=3, resolution=300)
    L.fit([valid])
    pl = L.transform([valid])[0].reshape(3, 300)
    grid = L.grid_

    for k in range(3):
        ax.plot(grid, pl[k], label=rf"$\lambda_{{{k+1}}}$")

    ax.set_xlim(0, 15)
    ax.set_title(f"Metric {m}")
    ax.set_xlabel("Filtration value")
    ax.legend()
    
# Hide unused subplots
for j in range(len(metrics), len(axes)):
    axes[j].axis("off")

axes[0].set_ylabel("Landscape value")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.12, 1))
plt.suptitle("Persistence Landscapes ($H_0$) for Metrics 1, 2, 3, 5, 8", y=1.05, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.savefig("landscape_subplots.png")
plt.close()
    



