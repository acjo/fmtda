"""Metric funtions."""

import numpy as np
from numpy.typing import NDArray
from pandas.core.config_init import is_int
from pandas.core.frame import DataFrame, Series

from fmtda._metric_base import euclidean, taxi_cab
from fmtda.parse_dict import get_abbrev_map

abbrev2desc, _ = get_abbrev_map()




def _extract_features(
    x: Series | DataFrame, feature_set: list[list[str]], transforms: list[str]
) -> NDArray:
    if isinstance(x, Series):
        features = [x.loc[feature].to_numpy() for feature in feature_set]
        features = [f.sum() if t == "sum" else f for f, t in zip(features, transforms)]
        return np.asarray(features)
    else:
        features = [x.loc[:, feature].to_numpy() for feature in feature_set]
        return np.column_stack(
            [
                f.sum(axis=1) if t == "sum" else f
                for f, t in zip(features, transforms)
            ]
        )


def metric_1(c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame):
    """First metric based on left and right side of the body."""
    left_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("left" == word.lower() for word in desc.split(" "))
    ]
    right_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("right" == word.lower() for word in desc.split(" "))
    ]

    group = [["gp"]]

    feature_set = group + [right_features] + [left_features]

    transforms = ["sum"] * len(feature_set)

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    return taxi_cab(c, x_vals, y_vals)


def metric_2(c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame):
    """second metric based on arms and legs."""
    arm_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("arm" == word.lower() for word in desc.split(" "))
    ]
    leg_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("leg" == word.lower() for word in desc.split(" "))
    ]

    group = [["gp"]]

    feature_set = group + [arm_features] + [leg_features]
    transforms = ["sum"] * len(feature_set)

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    return taxi_cab(c, x_vals, y_vals)


def metric_3(c: np.ndarray, x: Series | DataFrame, y: DataFrame | Series):
    """First metric based on upper and lower part of the body."""
    upper_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("upper" == w.lower() for w in desc.split(""))
    ]
    lower_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("lower" == w.lower() for w in desc.split(""))
    ]

    group = [["gp"]]

    feature_set = group + [upper_features] + [lower_features]
    transforms = ["sum"] * len(feature_set)

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(x, feature_set, transforms)

    return taxi_cab(c, x_vals, y_vals)


def metric_4():
    return


def metric_5():
    return


def metric_6():
    return


def metric_7():
    return


def metric_8():
    return


def metric_9():
    return


def metric_10():
    return


def metric_11():
    return


def metric_12():
    return


def metric_13():
    return


def metric_14():
    return


def metric_15():
    return


def metric_16():
    return


def metric_17():
    return


def metric_18():
    return


def metric_19():
    return
