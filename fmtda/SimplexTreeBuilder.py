"""Simplex Tree for TDA analysis."""

from typing import Optional

import gudhi
import matplotlib.pyplot as plt
import numpy as np


class SimplexTreeBuilder:
    """GUDHI SimplexTree.

    Builds a Guidhi Simplex tree with the following options:

    Parameters
    ----------
    points : list
        List of points (if constructing a Delaunay-based or Rips complex)
    filtration : str
        type of filtration to construct. Options include:

            * "rips",
            * "alpha",
            * "cech",
            * "delaunay"

    max_edge_length: float, Optional
        Threshold for Rips edges.
    max_alpha_square: float, Optional
        Threshold for alpha-based filtrations (squared radius).
    distance_matrix: ndarray
        Precomputed distances (for Rips only).
    precision: str
        Preciscion for Delaunay-based methods:

            * "fast",
            * "safe",
            * "exact"

    """

    def __init__(
        self,
        points: Optional[list] = None,
        filtration_type: str = "rips",
        max_edge_length: Optional[float] = None,
        max_alpha_square: Optional[float] = None,
        distance_matrix: Optional[np.ndarray] = None,
        precision: str = "safe",
    ) -> None:
        self.points = points
        self.filtration_type = filtration_type.lower()
        self.max_edge_length = max_edge_length
        self.max_alpha_square = max_alpha_square
        self.distance_matrix = distance_matrix
        self.precision = precision

    def build_simplex_tree(
        self, max_dimension: int = 3, output_squared_values: bool = True
    ):
        """Build a simplex tree.

        Parameters
        ----------
        max_dimension : int
            Maximum allowed dimension for p-simplices
        output_squared_values : bool
            Whether or not to output squared values

        """
        if self.filtration_type == "rips":
            # Rips Complex from points or a distance matrix
            if self.max_edge_length is None:
                raise ValueError("Rips requires max_edge_length.")
            if self.distance_matrix is not None:
                rips_complex = gudhi.RipsComplex(
                    distance_matrix=self.distance_matrix,
                    max_edge_length=self.max_edge_length,
                )
            elif self.points is not None:
                rips_complex = gudhi.RipsComplex(
                    points=self.points, max_edge_length=self.max_edge_length
                )
            else:
                raise ValueError(
                    "Provide either points or distance_matrix for Rips."
                )
            simplex_tree = rips_complex.create_simplex_tree(
                max_dimension=max_dimension
            )

        # Truthfully I've only tried this out with Rips and know it works for Rips
        elif self.filtration_type == "alpha":
            # Alpha Complex from a Delaunay triangulation
            if self.points is None:
                raise ValueError("AlphaComplex requires points.")
            alpha_complex = gudhi.AlphaComplex(
                points=self.points, precision=self.precision
            )
            simplex_tree = alpha_complex.create_simplex_tree(
                max_alpha_square=self._get_alpha_sq()
            )

        elif self.filtration_type == "cech":
            # Delaunay Cech Complex from a Delaunay triangulation
            if self.points is None:
                raise ValueError("DelaunayCechComplex requires points.")
            cech_complex = gudhi.DelaunayCechComplex(
                points=self.points, precision=self.precision
            )
            simplex_tree = cech_complex.create_simplex_tree(
                max_alpha_square=self._get_alpha_sq(),
                output_squared_values=output_squared_values,
            )

        elif self.filtration_type == "delaunay":
            # Delaunay Complex with no filtration
            if self.points is None:
                raise ValueError("DelaunayComplex requires points.")
            delaunay_complex = gudhi.DelaunayComplex(
                points=self.points, precision=self.precision
            )
            simplex_tree = delaunay_complex.create_simplex_tree(filtration=None)

        else:
            raise ValueError(
                "filtration_type must be 'rips', 'alpha', 'cech', or 'delaunay'."
            )

        self.simplex_tree = simplex_tree

        return simplex_tree

    def _get_alpha_sq(self):
        return (
            self.max_alpha_square
            if self.max_alpha_square is not None
            else float("inf")
        )

    def get_betti_numbers(self):
        if not hasattr(self, "simplex_tree"):
            raise RuntimeError(
                "Simplex tree has not been built yet. Call build_simplex_tree() first."
            )

        self.simplex_tree.compute_persistence()
        return self.simplex_tree.betti_numbers()

    def plot_persistence_diagram(self, legend=True):
        if not hasattr(self, "simplex_tree"):
            raise RuntimeError("Simplex tree has not been built yet.")

        self.simplex_tree.compute_persistence()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, legend=legend)
        plt.title("Persistence Diagram")
        plt.show()

    def plot_persistence_barcode(self):
        if not hasattr(self, "simplex_tree"):
            raise RuntimeError("Simplex tree has not been built yet.")

        self.simplex_tree.compute_persistence()
        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_barcode(diag)
        plt.title("Persistence Barcode")
        plt.show()
