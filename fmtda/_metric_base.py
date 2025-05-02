"""metric functions."""

import numpy as np


def taxi_cab(c: np.ndarray, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
    """Taxi-cab geometry metric.

    If x and y are matrices (2d arrays), we expect each row to be a datapoint.

    Parameters
    ----------
    c : np.ndarray
        Weighting constant that scales individual pieces of the metric.
        These must be positive values.

    x : np.ndarray
        vector of 1 point, or matrix of multiple points. In the case of a matrix each row should be a point.
    y : np.ndarry
        vector of 1 point, or matrix of multiple points. In the case of a matrix each row should be a point.

    Raises
    ------
    ValueError
        * If c has non-positive values.
        * If x or y has more than 2 dimensions.
    """
    if (c < 0).any():
        raise ValueError(
            "Constant vector needs to be elementwise non-negative."
        )
    if (c == 0).sum() == c.size:
        raise ValueError(
            "Constant vector needs to contain at least one strictly positive value."
        )

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if x.ndim != 2:
        raise ValueError("point array x cannot have greater than 2 dimensions.")
    if y.ndim != 2:
        raise ValueError("point array y cannot have greater than 2 dimensions.")
    d = (c * np.abs(x - y)).sum(axis=1)

    if d.size == 1:
        d = d.item()
    return d


def euclidean(c: float, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
    """Euclidean geometry metric.

    If x and y are matrices (2d arrays), we expect each row to be a datapoint.

    Parameters
    ----------
    c : float
        Weighting constant that scales individual the metric.

    x : np.ndarray
        vector of 1 point, or matrix of multiple points. In the case of a matrix each row should be a point.
    y : np.ndarry
        vector of 1 point, or matrix of multiple points. In the case of a matrix each row should be a point.

    Raises
    ------
    ValueError
        * If c is non-positive or if c is not a float.
        * If x or y has more than 2 dimensions.
    """
    if not isinstance(c, float):
        raise ValueError("Constant scale needs to be a float.")
    if c <= 0:
        raise ValueError("Constant value needs to be elementwise postive.")

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if x.ndim != 2:
        raise ValueError("point array x cannot have greater than 2 dimensions.")
    if y.ndim != 2:
        raise ValueError("point array y cannot have greater than 2 dimensions.")

    d = c * np.linalg.norm(x - y, axis=1)
    if d.size == 1:
        d = d.item()
    return d
