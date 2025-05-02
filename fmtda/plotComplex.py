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
rcParams["figure.dpi"] = 300
rcParams["axes.titlesize"] = 15


def build_nx_graph(
    complex: SimplicialComplex, data: np.ndarray
) -> tuple[nx.Graph, list[str]]:
    G = nx.Graph()

    for s, p_simplicies in complex.simplices.items():
        for simplex in p_simplicies:
            val = list(simplex)
            if s == 0:
                G.add_node(val[0])
            elif s == 1:
                G.add_edge(val[0], val[1])

    clusters = nx.connected_components(G)
    node_types = {}
    for cluster in clusters:
        cluster_list = list(cluster)
        have_fibro = data[cluster_list, 0].astype(bool)
        print(f"Number of points in cluster: {len(cluster_list)}")
        print(
            f"Percentage of points that have fibromylagia: {have_fibro.sum().item()/have_fibro.size}"
        )
        for node, partition_bool in zip(cluster_list, have_fibro):
            if partition_bool:
                node_types[node] = "red"
            else:
                node_types[node] = "blue"

    node_colors: list[str] = [node_types[node] for node in G.nodes()]

    return G, node_colors


def plot_nx_graph(
    G: nx.Graph, colors: list[str], title: str = "", labels: bool = True
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    nx.draw_networkx(
        G,
        pos=nx.nx_agraph.graphviz_layout(G),
        node_size=150,
        font_size=8,
        with_labels=True,
        font_weight="bold",
        node_color=colors,
        labels={node: "" for node in G.nodes()},
        ax=ax,
    )

    color_dict = {"Fibro": "red", "Control": "blue"}
    if labels:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=node_type)
            for node_type, color in color_dict.items()
        ]
        ax.legend(handles=legend_elements, loc="best")
    plt.tight_layout()

    return fig


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
    np.concatenate((np.random.random(1), np.ones(2))),  # 1
    np.concatenate((np.random.random(1), np.ones(2))),  # 2
    np.concatenate((np.random.random(1), np.ones(2))),  # 3
    np.random.random(1),  # 4
    np.concatenate((np.random.random(1), np.random.random(1))),  # 5
    np.random.random(2),  # 6
    np.random.random(1),  # 7
    np.random.random(2),  # 8
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


def gen_figures(metric_idx, thresholds):
    metric = metrics[metric_idx]
    data = data_collection[metric_idx]
    D = metric.dist_matrix(data)
    for r in thresholds:
        rho = np.array([r], dtype=float)
        filtration = SimplicialComplex.build_filtration_incrementally(D, rho)
        graph, colors = build_nx_graph(filtration[0], data)
        # colors = ["blue" for _ in graph.nodes()]
        title = f"Clusters for metric {metric_idx} and radius {rho.item():.1f}"

        file_name = f"metric_{metric_idx}_{r:.1f}.png"
        plot_nx_graph(graph, colors, title=title, labels=True)
        plt.savefig(file_name)
        plt.close(plt.gcf())


if __name__ == "__main__":
    # for cc in constant_arrays:
    #     print(cc)
    # print("Metric 5:")
    # metric_idx = 5
    # thresholds = np.array([2.0, 5.2, 13, 14], dtype=float)
    # gen_figures(metric_idx, thresholds)
    print("Metric 8:")
    metric_idx = 8
    thresholds = np.array([13.0])
    gen_figures(metric_idx, thresholds)
    print()
