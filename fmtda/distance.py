"""metric module."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from pandas.core.frame import DataFrame, Series
from scipy.spatial.distance import cdist

from fmtda import metric_types


class Metric(object):
    """Metric class.

    Parameters
    ----------
    t : int
        Integer that indicates the type of metric
    c : (n,) ndarray
        The weighting vector for the metric.

    Attributes
    ----------
    type : int
        stores the t parameter
    c : (n,) ndarray
        stores the c parameter
    fn : Callable
        the metric Callable
    __call__ : Callable
        returns the distance between x and y useing self.fn
    dist_matrix : Callable
        Returns the distance matrix using self.fn

    """

    def __init__(self, t: int, c: NDArray) -> None:
        if not isinstance(t, int):
            raise ValueError(f"type parameter must be int, not {type(t)}")
        if not isinstance(c, np.ndarray):
            raise ValueError(
                f"Constant multiple parameter must be an np.ndarray, not {type(c)}"
            )
        self.type: int = t
        self.c: NDArray = c
        fn_name = f"metric_{self.type}"
        if hasattr(metric_types, fn_name):
            self.fn: Callable = eval(f"metric_types.{fn_name}")
        else:
            raise AttributeError(
                f"fmtda.metric_types does not have a function {fn_name}."
            )
        return

    def __call__(
        self, x: DataFrame | Series, y: DataFrame | Series
    ) -> float | NDArray:
        """Calculate the distance between x and y.

        Parameters
        ----------
        x : (N,M) DataFrame or (,M) Series
            first point. If 2D rows contain points and columns contain features
        y : (N,M) DataFrame or (,M) Series
            Second point. If 2D rows contain points and columns contain features

        x and y have to both be 1d or both be 2d.

        Returns
        -------
        distance : float or ndarray
            Distance betwen the points. If x and y are 2d (N,M) ndarrays the result will be a (,N) ndarray of distances.
        """
        return self.fn(self.c, x, y)

    def dist_matrix(self, X: DataFrame) -> NDArray:
        """Return the distance matrix.

        Parameters
        ----------
        X : (N,M) DataFrame
            Array of points

        Returns
        -------
        D : (N,N) ndarray
            NxN array of distances
        """
        return cdist(X, X, metric=self)
