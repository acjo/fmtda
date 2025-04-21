"""metric module."""

from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from scipy.spatial.distance import cdist

from fmtda.metric_types import *
from fmtda.parse_dict import get_abbrev_map


class Metric(object):
    def __init__(self, type: int, c: NDArray) -> None:
        self.type: int = type
        self.c: NDArray = c
        fn_name = f"metric_{self.type}"
        self.fn: Callable = eval(fn_name)
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
        D = cdist(X, X, metric=self)
        return D
