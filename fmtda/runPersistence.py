import gudhi
import numpy as np
import pandas as pd

from fmtda import Metric, SimplexTreeBuilder

patientData = pd.read_excel("fmtda/Clinical_fm_66_.xlsx", sheet_name="data_66")
constant_arrays = [np.ones(1)]

for i, c in enumerate(constant_arrays):
    metric = Metric(i + 1, c)

    distMat = metric.dist_matrix(patientData)

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist) / 8
    thresholds = np.arange(minDist, maxDist + interval, interval)

    treeForThisMetric = SimplexTreeBuilder(None, "rips", distMat, "safe")
    for m in range(len(thresholds)):
        addTreeForThisFiltValue = treeForThisMetric.build_simplex_tree(
            thresholds[m], 3, False
        )

    treeForThisMetric.plot_persistence_diagram(True)
