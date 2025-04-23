from pathlib import Path

import gudhi as gd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import cdist

from fmtda import Metric, SimplexTreeBuilder
from fmtda import utils

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

# set weights for metrics.
constant_arrays = [
    np.random.random(3),
    np.random.random(3),
    np.random.random(3),
    np.random.random(1),
    np.random.random(2),
    np.random.random(2),
    np.random.random(1),
    np.random.random(2),
]
# constant_arrays.append(constant_arrays)
# feature_set = []

for i, c in enumerate(constant_arrays):
    # instantiate metric
    metric = Metric(i + 1, c)
    print(f"Metric: {i+1}")

    feature_fun = eval(f"utils.feature_{i + 1}")

    x_vals, feature_set, transforms = feature_fun(patientData)

    x_df = pd.DataFrame(data=x_vals)
    x_df.dropna(axis=0, inplace=True)

    data = x_df.values
    print(data.shape)
    distMat = metric.dist_matrix(data)

    print(f"Dist matrix shape: {distMat.shape}")

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist) / 8

    print(f"minDist: {minDist}")
    print(f"maxDist: {maxDist}")
    print(f"interval: {interval}")

    st_rips = gd.RipsComplex(
        distance_matrix=distMat, max_edge_length=np.inf
    ).create_simplex_tree(max_dimension=2)

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
    gd.plot_persistence_diagram(diagram_rips, alpha=0.3)
    _ = plt.title("Persistence diagram of the Rips complex")

    plt.show()
