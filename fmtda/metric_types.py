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


def metric_1(c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame) -> float:
    """First metric based on left and right side of the body.
    
    Parameters
    ----------
    c : NDArray
        Constant weight factory array
    x : Series or DataFrame
        x point
    y : Series or DataFrame
        y point

    Returns
    -------
    d : float
        distance
    """
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


def metric_2(c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame) -> float:
    """Second metric based on arms and legs.
    
    Parameters
    ----------
    c : NDArray
        Constant weight factory array
    x : Series or DataFrame
        x point
    y : Series or DataFrame
        y point

    Returns
    -------
    d : float
        distance
    """
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


def metric_3(c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame) -> float:
    """Third metric based on upper and lower part of the body.
    
    Parameters
    ----------
    c : NDArray
        Constant weight factory arraa
    x : Series or DataFrame
        x point
    y : Series or DataFrame
        y point

    Returns
    -------
    d : float
        distance
    """
    upper_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("upper" == word.lower() for word in desc.split(" "))
    ]
    lower_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("lower" == word.lower() for word in desc.split(" "))
    ]

    group = [["gp"]]

    feature_set = group + [upper_features] + [lower_features]
    transforms = ["sum"] * len(feature_set)

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    return taxi_cab(c, x_vals, y_vals)


def metric_4(c:NDArray, x:Series | DataFrame, y: Series | DataFrame) -> float:
    """Metric 4 based off the type of pain.
    
    Parameters
    ----------
    c : NDArray
        Constant weight factory arraa
    x : Series or DataFrame
        x point
    y : Series or DataFrame
        y point

    Returns
    -------
    d : float
        distance
    """

    if not c.size == 1:
        raise RuntimeError("c must have only one element.")

    group = [["gp"]]

    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc().items()
        if any("pain" == word.lower() for word in desc.split(" ") 
               and all("psychological" != word.lower() for word in desc.split(" ")))

    ]

    feature_set = group + [pain_features]
    transforms = ["sum"] * len(feature_set)

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    d1 = c[0] * np.abs(x_vals[0] - y_vals[0])
    d2 = euclidean(1, x_vals[1:], y_vals[1:])
    d = d1 + d2

    return d



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
