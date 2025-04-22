"""Metric funtions."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from pandas.core.array_algos import transforms
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
        features = [
            f.sum() if t == "sum" else f for f, t in zip(features, transforms)
        ]
        return np.asarray(features).squeeze()
    else:
        raise RuntimeError("vector required to be a series, not a DataFrame.")
        # features = [x.loc[:, feature].to_numpy() for feature in feature_set]
        # return np.column_stack(
        #     [
        #         f.sum(axis=1) if t == "sum" else f
        #         for f, t in zip(features, transforms)
        #     ]
        # )


def metric_1(
    c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame
) -> float:
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

    d = taxi_cab(c, x_vals, y_vals)
    d = cast(float, d)
    return d


def metric_2(
    c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame
) -> float:
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

    d = taxi_cab(c, x_vals, y_vals)
    d = cast(float, d)
    return d


def metric_3(
    c: np.ndarray, x: Series | DataFrame, y: Series | DataFrame
) -> float:
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

    d = taxi_cab(c, x_vals, y_vals)
    d = cast(float, d)
    return d


def metric_4(c: NDArray, x: Series | DataFrame, y: Series | DataFrame) -> float:
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

    group = ["gp"]

    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("pain" == word.lower() for word in desc.split(" "))
        and all("psychological" != word.lower() for word in desc.split(" "))
    ]
    pain_features.remove("14_")

    feature_set = [group + pain_features]
    transforms = ["identity"]

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    d0 = c[0] * np.abs(x_vals[0] - y_vals[0])
    d1 = euclidean(1.0, x_vals[1:], y_vals[1:])
    d = d0 + d1
    d = cast(float, d)

    return d


def metric_5(c: NDArray, x: Series | DataFrame, y: Series | DataFrame) -> float:
    """Metric 5 based off height and regions where pain is present.

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
    if not c.size == 2:
        raise RuntimeError("c must have exactly two elements.")

    regions_of_pain = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any(
            "arm" == word.lower()
            or "leg" == word.lower()
            or "upper" == word.lower()
            or "lower" == word.lower()
            or "left" == word.lower()
            or "right" == word.lower()
            for word in desc.split(" ")
        )
    ]
    feature_set = [["gp", "16_h"] + regions_of_pain]
    transforms = ["identity"]
    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    d0 = taxi_cab(c, x_vals[:2], y_vals[:2])
    d1 = euclidean(1.0, x_vals[2:], y_vals[2:])
    d = d0 + d1
    d = cast(float, d)

    return d


def metric_6(c: np.ndarray, x: Series, y: Series) -> float:
    """Metric 6 based off BMI and types of pain.

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
    group = ["gp"]
    bmi = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if desc == "body mass index"
    ]
    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("pain" == word.lower() for word in desc.split(" "))
    ]
    pain_features.remove("14_")

    feature_set = [group + bmi + pain_features]
    transforms = ["identity"]

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)
    d0 = taxi_cab(c, x_vals[:2], y_vals[:2])
    d1 = euclidean(1.0, x_vals[2:], y_vals[2:])

    d = d0 + d1
    d = cast(float, d)

    return d


def metric_7(c: np.ndarray, x: Series, y: Series) -> float:
    """Metric 7 based off mental health systems and types of pain.

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
        raise RuntimeError("c must have exactly one element.")

    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("pain" == word.lower() for word in desc.split(" "))
    ]
    pain_features.remove("14_")

    feature_set = [["gp", "13_g_15", "13_lw_15", "3_sss_11"] + pain_features]
    transforms = ["identity"]
    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)

    d0 = taxi_cab(c, x_vals[:1], y_vals[:1])
    d1 = taxi_cab(np.ones_like(x_vals[1:]), x_vals[1:], y_vals[1:])
    d = d0 + d1
    d = cast(float, d)
    return d


def metric_8(c: np.ndarray, x: Series, y: Series) -> float:
    """Metric 6 based off gastro intenstenial symptoms and regions where pain is present.

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
    group = ["gp"]
    gastro = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if desc == "gastrointestinal symptoms"
    ]
    regions_of_pain = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any(
            "arm" == word.lower()
            or "leg" == word.lower()
            or "upper" == word.lower()
            or "lower" == word.lower()
            or "left" == word.lower()
            or "right" == word.lower()
            for word in desc.split(" ")
        )
    ]

    feature_set = [group + gastro + regions_of_pain]
    transforms = ["identity"]

    x_vals = _extract_features(x, feature_set, transforms)
    y_vals = _extract_features(y, feature_set, transforms)
    d0 = taxi_cab(c, x_vals[:2], y_vals[:2])
    d1 = taxi_cab(np.ones_like(x_vals[2:]), x_vals[2:], y_vals[2:])

    d = d0 + d1
    d = cast(float, d)
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
