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
    if isinstance(c, float):
        c = np.array([c], dtype=float)
    if c.size != 1:
        raise RuntimeError("c must be a float or a single element ndarray.")

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)
    d0 = c[0] * np.abs(x_2d[:, 0] - y_2d[:, 0])
    d1 = euclidean(1.0, x_2d[:, 1:], y_2d[:, 1:])
    d = d0 + d1

    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
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

    full_c = np.concatenate((c, np.ones(x_2d.shape[1] - 2, dtype=float)))

    d = taxi_cab(full_c, x_2d, y_2d)

    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()

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
    if not c.size == 2:
        raise RuntimeError("c must have exactly two elements.")

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)

    d0 = taxi_cab(c, x_2d[:, :2], y_2d[:, :2])
    d1 = euclidean(1.0, x_2d[:, 2:], y_2d[:, 2:])

    d = d0 + d1

    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()

    return d


def metric_7(c: NDArray, x: Series, y: Series) -> float | NDArray:
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
    if isinstance(c, float):
        c = np.array([c], dtype=float)
    if c.size != 1:
        raise RuntimeError("c must be a float or a single element ndarray.")

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)

    full_c = np.concatenate((c, np.ones(3, dtype=float)))

    d0 = taxi_cab(full_c, x_2d[:, :4], y_2d[:, :4])
    d1 = euclidean(1.0, x_2d[:, 4:], y_2d[:, 4:])
    d = d0 + d1
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return d


def metric_8(c: NDArray, x: Series, y: Series) -> float | NDArray:
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
    if not c.size == 2:
        raise RuntimeError("c must have exactly two elements.")

    x_2d = np.atleast_2d(x)
    y_2d = np.atleast_2d(y)

    full_c = np.concatenate((c, np.ones(x_2d.shape[1] - 2, dtype=float)))

    d = taxi_cab(full_c, x_2d, y_2d)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()

    return d


def metric_9(w:NDArray, c: list[NDArray], x: list[NDArray], y: list[NDArray]) -> float | NDArray:
    """
    Metric 8 based on a weighted combination of metrics 1 - 8.

    Note: The X and Y inputs do not take into account the way the new data preprocessing works.
    Please delete this comment when updated.

    Parameters
    ----------
    w : NDArray
        The weights for the weighted sum of the metrics.
    c : NDArray
        Constant weight factory array
    x : NDarray
        x point
    y : NDArray
        y point

    Returns
    -------
    d : float
        distance
    """
    if len(c) == 8:
        raise RuntimeError(
            "c must have eight elements individual weights arrays for each metric."
        )

    d = [eval(f"metric_{i}")(c[0], x[i], y[i]) for i in range(1, 9)]

    if x.ndim == 1:
        d = float(np.inner(w, np.asarray(d)))

    else:
        d = (w * np.column_stack(d)).sum(axis=1)

    return d
