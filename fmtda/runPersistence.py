import pandas as pd
import numpy as np
import gudhi
from SimplexTreeBuilder import SimplexTreeBuilder

patientData = pd.read_excel('fmtda/Clinical_fm_66_.xlsx', sheet_name='data_66')

for i in range(13): #number of metrics
    distMat = np.zeros((len(patientData), len(patientData)))
    for j in range(len(patientData)):
        for k in range(j + 1, len(patientData)):
            distMat[j, k] = 2 #placeholder for distanceFunction(patientData[j,:], patientData[k,:])

    minDist = np.min(distMat)
    maxDist = np.max(distMat)
    interval = (maxDist - minDist)/8
    thresholds = np.arange(minDist, maxDist + interval, interval)

    treeForThisMetric = SimplexTreeBuilder(None,"rips",distMat,"safe")
    for m in range(len(thresholds)):
        addTreeForThisFiltValue = treeForThisMetric.build_simplex_tree(thresholds[m],3,False)
    
    treeForThisMetric.plot_persistence_diagram(True)

