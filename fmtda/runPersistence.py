from pathlib import Path

import gudhi as gd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import cdist

from fmtda import Metric, SimplexTreeBuilder

rcParams["font.family"] = "serif"
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.figsize"] = (7, 7)
rcParams["figure.dpi"] = 200
rcParams["axes.titlesize"] = 15


data_path = Path(__file__).parent / "Clinical_fm_66_.xlsx"
np.random.seed(32)
patientData = pd.read_excel(data_path, sheet_name="data_66")
# patientData = patientData.loc[:, ALL_FEATURES]
# patientData.dropna(axis=0, inplace=True)
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

for i, c in enumerate(constant_arrays):
    print(f"metric: {i+1}")
    metric = Metric(i + 1, c)

    distMat = metric.dist_matrix(patientData)
    print(f"Dist mat shape: {distMat.shape}")
    print(f"Total elements: {distMat.size}")
    print(f"How many nans: {np.isnan(distMat).sum()}")

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist) / 8
    print(f"minDist: {minDist}")
    print(f"maxDist: {maxDist}")
    print(f"interval: {interval}")
    # thresholds = np.arange(minDist, maxDist + interval, interval)

    st_rips = gd.RipsComplex(
        distance_matrix=distMat, max_edge_length=1.55
    ).create_simplex_tree(max_dimension=2)

    rips_filtration = st_rips.get_filtration()
    rips_list = list(rips_filtration)
    for splx in rips_list[300:600]:
        print(splx)

    thresholds = [t for _, t in rips_list]

    close_to_1_55 = np.argmin(np.abs(np.asarray(thresholds) - 1.55))

    print(f"connected compoentent close to 1.55: {rips_list[close_to_1_55]}")

    diagram_rips = st_rips.persistence(
        homology_coeff_field=2, persistence_dim_max=True, min_persistence=1.45
    )
    gd.plot_persistence_diagram(diagram_rips, alpha=0.3)
    _ = plt.title("Persistence diagram of the Rips complex")

    # treeForThisMetric = SimplexTreeBuilder(None, "rips", distMat, "safe")
    # for m in range(len(thresholds)):
    #     treeForThisMetric.build_simplex_tree(thresholds[m], 3, False)

    # treeForThisMetric.plot_persistence_diagram(True)
    plt.show()
