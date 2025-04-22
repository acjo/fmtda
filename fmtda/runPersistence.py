import numpy as np
import pandas as pd
from pathlib import Path
from fmtda import Metric, SimplexTreeBuilder
from fmtda import ALL_FEATURES
from matplotlib import rcParams
from matplotlib import pyplot as plt

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
patientData = pd.read_excel(data_path, sheet_name="data_66")
# patientData = patientData.loc[:, ALL_FEATURES]
patientData.dropna(axis=0, inplace=True)
constant_arrays = [
    np.ones(3),
    np.ones(3),
    np.ones(3),
    np.ones(1),
    np.ones(2),
]

for i, c in enumerate(constant_arrays):
    print(f"metric: {i+1}")
    metric = Metric(i + 1, c)

    distMat = metric.dist_matrix(patientData)
    print("Complete!")
    continue

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist) / 8
    print(f"minDist: {minDist}")
    print(f"maxDist: {maxDist}")
    print(f"interval: {interval}")
    thresholds = np.arange(minDist, maxDist + interval, interval)

    treeForThisMetric = SimplexTreeBuilder(None, "rips", distMat, "safe")
    for m in range(len(thresholds)):
        addTreeForThisFiltValue = treeForThisMetric.build_simplex_tree(
            thresholds[m], 3, False
        )

    treeForThisMetric.plot_persistence_diagram(True)
    plt.show()
