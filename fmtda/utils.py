"""utility functions"""

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from pandas.core.flags import NDFrame

from fmtda.parse_dict import get_abbrev_map

abbrev2desc, _ = get_abbrev_map()

ALL_FEATURES = ["gp"]
for abbrev, desc in abbrev2desc.items():
    for word in desc.lower().split(" "):
        if (
            word == "left"
            or word == "right"
            or word == "arm"
            or word == "leg"
            or word == "gp"
            or word == "upper"
            or word == "lower"
        ):
            ALL_FEATURES.append(abbrev)
            break


def extract_features(
    x: Series | DataFrame, feature_set: list[list[str]], transforms: list[str]
) -> NDArray:
    """Extract features.

    Parameters
    ----------
    x : Series or DataFrame
        The pandas collection holding the data
    feature_set : list of lists of strings
        The feature set. Each sublist is a collection of features to apply a transform function to.
    transforms : list[str]
        The transform functions to apply to each sub list in feature_set.
        if "x" is a DataFrame "sum" will sum across the columns.

    Returns
    -------
    y : np.array
        the array with the extracted features
    """
    if isinstance(x, Series):
        features = [x.loc[feature].to_numpy() for feature in feature_set]
        features = [
            f.sum() if t == "sum" else f for f, t in zip(features, transforms)
        ]
        return np.asarray(features).squeeze()
    else:
        features = [x.loc[:, feature].to_numpy() for feature in feature_set]
        features = [
            f.sum(axis=1) if t == "sum" else f
            for f, t in zip(features, transforms)
        ]
        return np.column_stack(features)


def feature_1(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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

    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_2(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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

    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_3(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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

    x_vals = extract_features(x, feature_set, transforms)

    return x_vals, feature_set, transforms


def feature_4(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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
    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_5(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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

    x_vals = extract_features(x, feature_set, transforms)

    return x_vals, feature_set, transforms


def feature_6(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
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
    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_7(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    return


def feature_8(
    x: Series | DataFrame,
) -> tuple[NDFrame, list[list[str]], list[str]]:
    return
