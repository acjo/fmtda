from copy import deepcopy
from pathlib import Path

import gudhi as gd  # type: ignore
import numpy as np
import pandas as pd
from gudhi.representations import Entropy
from gudhi.representations.vector_methods import Landscape
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
rcParams["figure.dpi"] = 200
rcParams["axes.titlesize"] = 15


# datapath
data_path = Path(__file__).parent / "Clinical_fm_66_.xlsx"
# set random seet for reproducability
np.random.seed(32)
# read patient data
patientData = pd.read_excel(data_path, sheet_name="data_66")
max_dim = 2
entropies = {}

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

    # rips_filtration = st_rips.get_filtration()
    # rips_list = list(rips_filtration)
    # for splx in rips_list[300:600]:
    #     print(splx)

    # thresholds = [t for _, t in rips_list]

    # close_to_1_55 = np.argmin(np.abs(np.asarray(thresholds) - 1.55))

    # print(f"connected compoentent close to 1.55: {rips_list[close_to_1_55]}")

    diagram_rips = st_rips.persistence(
        homology_coeff_field=2, persistence_dim_max=True, min_persistence=1.45
    )

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

            # Plotting Persistence Landscapes for H1
            if d == 1:
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
