"""Metric funtions."""

from optparse import Values
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pandas.core.array_algos import transforms
from pandas.core.config_init import is_int
from pandas.core.frame import DataFrame, Series

from fmtda._metric_base import euclidean, taxi_cab
from fmtda.parse_dict import get_abbrev_map

abbrev2desc, _ = get_abbrev_map()


def _input_check(c: NDArray, x: NDArray, y: NDArray) -> None:
    """Input checks to metrics."""
    if x.shape != y.shape:
        raise RuntimeError(
            f"x and y expected to be the same shape.\nShape of x: {x.shape}, shape of y: {y.shape}"
        )
    elif x.ndim == 1:
        if c.size != x.size:
            raise RuntimeError(
                "Feature vector is not the same size as the weighting vector."
            )
    elif x.ndim == 2:
        if c.size != x.shape[1]:
            raise RuntimeError(
                "More features than weighting constant, parameter c expected to have the same length as the number of columns of x and y."
            )
    elif x.ndim > 2:
        raise RuntimeError("x expected to have 1 or 2 dimensions.")
    elif y.ndim > 2:
        raise RuntimeError("y expected to have 1 or 2 dimensions.")

    return


def metric_1(c: NDArray, x: NDArray, y: NDArray) -> float | NDArray:
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

    Raises
    ------
    ValueError
        * if x and y are not the same shape
        * If c is not the right size (c has to be the same size as the number of features of x and y)
        * if the number of dimensions of x/y exceeds 2.
    """
    _input_check(c, x, y)
    d = taxi_cab(c, x, y)
    return d


def metric_2(c: NDArray, x: NDArray, y: NDArray) -> float | NDArray:
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

    Raises
    ------
    ValueError
        * if x and y are not the same shape
        * If c is not the right size (c has to be the same size as the number of features of x and y)
        * if the number of dimensions of x/y exceeds 2.
    """
    return metric_1(c, x, y)


def metric_3(c: NDArray, x: NDArray, y: NDArray) -> float | NDArray:
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

    Raises
    ------
    ValueError
        * if x and y are not the same shape
        * If c is not the right size (c has to be the same size as the number of features of x and y)
        * if the number of dimensions of x/y exceeds 2.
    """
    return metric_1(c, x, y)


def metric_4(c: NDArray | float, x: NDArray, y: NDArray) -> float | NDArray:
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
    if isinstance(c, np.ndarray) and c.size != 1:
        raise RuntimeError("c must have only one element if it is an array.")
    if isinstance(c, float):
        c = np.array([c], dtype=float)
    else:
        raise RuntimeError("c must be a float or a single element array.")

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)
    d0 = c[0] * np.abs(x_2d[:, 0] - y_2d[:, 0])
    d1 = euclidean(1.0, x_2d[:, 1:], y_2d[:, 1:])
    d = d0 + d1

    if x.ndim == 1:
        d = d.item()
    else:
        d = d.squeeze()
    return d


def metric_5(c: NDArray, x: NDArray, y: NDArray) -> float | NDArray:
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

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)
    d0 = taxi_cab(c, x_2d[:, :2], y_2d[:, :2])
    d1 = taxi_cab(np.ones_like(x_2d[0, 2:]), x_2d[:, 2:], y_2d[:, 2:])
    d = d0 + d1

    if isinstance(d, float):
        return d
    return d


def metric_6(c: NDArray, x: NDArray, y: NDArray) -> float | NDArray:
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
    return


def metric_7(c: NDArray, x: Series, y: Series) -> float:
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


def metric_8(c: NDArray, x: Series, y: Series) -> float:
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


def metric_13(c: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    
    """
    Metric 8 based on a weighted combination of metrics 1 - 8.
    
    Note: The X and Y inputs do not take into account the way the new data preprocessing works. 
    Please delete this comment when updated. 
    
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
    if not c.size == 8:
        raise RuntimeError("c must have eight elements.")
    
    d1 = metric_1(c[0], X[:,0], Y[:,0])
    d2 = metric_2(c[1], X[:,1], Y[:,1])
    d3 = metric_3(c[2], X[:,2], Y[:,2])
    d4 = metric_4(c[3], X[:,3], Y[:,3])
    d5 = metric_5(c[4], X[:,4], Y[:,4])
    d6 = metric_6(c[5], X[:,5], Y[:,5])
    d7 = metric_7(c[6], X[:,6], Y[:,6])
    d8 = metric_8(c[7], X[:,7], Y[:,7])
    
    d = np.sum(d1, d2, d3, d4, d5, d6, d7, d8)
    
    return d
    
    
    
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
