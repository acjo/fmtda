# %%
# import statements
from copy import deepcopy
from pathlib import Path

import gudhi as gd  # type: ignore
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from topocore import SimplicialComplex

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


# %%
# set up metric dependencies

# datapath
data_path = "Clinical_fm_66_.xlsx"
# set random seet for reproducability
np.random.seed(32)
# read patient data
patientData = pd.read_excel(data_path, sheet_name="data_66")

# set weights for metrics.
constant_arrays = [
    np.concatenate((np.random.random(1), np.ones(2))),
    np.concatenate((np.random.random(1), np.ones(2))),
    np.concatenate((np.random.random(1), np.ones(2))),
    np.random.random(1),
    np.concatenate((np.random.random(1) * 0.00001, np.random.random(1))),
    # np.random.random(2),
    np.random.random(2),
    np.random.random(1),
    np.random.random(2),
]
w = np.random.random(8)
copy_constant_arrays = deepcopy(constant_arrays)
constant_arrays.append((w, copy_constant_arrays))  # type: ignore
sizes = []
metrics = {}
feature_sets = {}
transform_collection = {}
data_collection = {}
# %% build metrics
for i, c in enumerate(constant_arrays):
    key = i + 1
    # instantiate metric
    metric = Metric(key, c)
    print(f"Metric: {key}")

    metrics[key] = metric

    feature_fun = eval(f"utils.feature_{key}")

    x_vals, feature_set, transforms = feature_fun(patientData)

    feature_sets[key] = feature_set
    transform_collection[key] = transforms

    x_df = pd.DataFrame(data=x_vals)
    x_df.dropna(axis=0, inplace=True)
    data = x_df.values

    if i != 9:
        sizes.append(data.shape[1])

    data_collection[key] = data

# %%
# plot metric 5 simplex at threshold value 5.5
metric_5 = metrics[5]
D = metric_5.dist_matrix(data_collection[5])

filtration = SimplicialComplex.build_filtration_incrementally(
    D, np.array([2], dtype=float)
)
G = filtration[0].visualize_complex()
fig = plt.gcf()
plt.savefig("metric_30_2.0.png")
plt.show()
# print(filtration[0].compute_homology_ranks())
H = filtration[0].find_homologies(0)
# print(H)
#

cc = list(nx.connected_components(G))
print(len(cc))

for c in cc:
    have_fibro = data_collection[5][list(c), 0].astype(bool)
    print(f"Number of components in cluster: {len(c)}")
    print(
        f"Percentage of the compoenents that have fibromyalaga:{sum(have_fibro)/len(have_fibro):.4f}"
    )
