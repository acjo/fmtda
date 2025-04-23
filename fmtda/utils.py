"""utility functions"""

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from fmtda.parse_dict import get_abbrev_map

abbrev2desc, _ = get_abbrev_map()


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
    """
    Features for first metric based on the left and right side of the body.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
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

    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_2(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for second metric based on arms and legs.
      left and right side of the body.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
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

    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_3(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for third metric based on the upper and lower parts of the body.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
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

    x_vals = extract_features(x, feature_set, transforms)

    return x_vals, feature_set, transforms


def feature_4(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for fourth metric based on the type of pain.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
    """
    group = ["gp"]

    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("pain" == word.lower() for word in desc.split(" "))
    ]

    pain_features.remove("14_")

    feature_set = [group + pain_features]
    transforms = ["identity"]
    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_5(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for fith metric based on height + regions where pain is present.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
    """
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
    """
    Features for 6th metric based on BMI and types of pain.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
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
    x_vals = extract_features(x, feature_set, transforms)
    return x_vals, feature_set, transforms


def feature_7(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for 7th metric based on mental health symptoms and types of pain.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
    """
    pain_features = [
        abbrev
        for abbrev, desc in abbrev2desc.items()
        if any("pain" == word.lower() for word in desc.split(" "))
    ]
    pain_features.remove("14_")

    feature_set = [["gp", "13_g_15", "13_lw_15", "3_sss_11"] + pain_features]
    transforms = ["identity"]
    x_vals = extract_features(x, feature_set, transforms)

    return x_vals, feature_set, transforms


def feature_8(
    x: Series | DataFrame,
) -> tuple[NDArray, list[list[str]], list[str]]:
    """
    Features for 8th metric based on Gastro intestinal symptoms and regions where pain is present.

    Parameters
    ----------
    x : Series or DataFrame
        Full dataset.

    Returns
    -------
    x_vals : NDArray
        feature matrix
    feature_set : list of list of strings
        each sublist contains the original features considered for a super feature
    transforms : list of strings
        transformation equation to turn the sub list of features into a super feature
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

    x_vals = extract_features(x, feature_set, transforms)

    return x_vals, feature_set, transforms
