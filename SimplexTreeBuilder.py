import gudhi
import pandas as pd
import matplotlib.pyplot as plt

class SimplexTreeBuilder:
    
    ''' Builds a GUDHI SimplexTree from one of the following:
      - 'rips': Rips complex (from points or a distance matrix),
      - 'alpha': Alpha complex,
      - 'cech': Delaunay Čech complex,
      - 'delaunay': Delaunay complex (no filtration). '''

    def __init__(self, 
                 points=None, 
                 filtration_type='rips',
                 max_edge_length=None,
                 max_alpha_square=None,
                 distance_matrix=None,
                 precision='safe'):
        
        ''' 
        Parameters:
            points: List of points (if constructing a Delaunay-based or Rips complex).
            filtration_type: Type of complex to construct ('rips', 'alpha', 'cech', or 'delaunay').
            max_edge_length: Threshold for Rips edges.
            max_alpha_square: Threshold for alpha-based filtrations (squared radius).
            distance_matrix: Precomputed distances (for Rips only).
            precision: 'fast', 'safe', or 'exact' (Delaunay-based methods).
        '''
        self.points = points
        self.filtration_type = filtration_type.lower()
        self.max_edge_length = max_edge_length
        self.max_alpha_square = max_alpha_square
        self.distance_matrix = distance_matrix
        self.precision = precision

    def build_simplex_tree(self, max_dimension=3, output_squared_values=True):
        
        if self.filtration_type == "rips":
            
            # Rips Complex from points or a distance matrix
            if self.max_edge_length is None:
                raise ValueError("Rips requires max_edge_length.")
            if self.distance_matrix is not None:
                rips_complex = gudhi.RipsComplex(
                    distance_matrix=self.distance_matrix,
                    max_edge_length=self.max_edge_length
                )
            elif self.points is not None:
                rips_complex = gudhi.RipsComplex(
                    points=self.points,
                    max_edge_length=self.max_edge_length
                )
            else:
                raise ValueError("Provide either points or distance_matrix for Rips.")
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
            
            
        # Truthfully I've only tried this out with Rips and know it works for Rips
        elif self.filtration_type == "alpha":
            
            # Alpha Complex from a Delaunay triangulation
            if not self.points:
                raise ValueError("AlphaComplex requires points.")
            alpha_complex = gudhi.AlphaComplex(points=self.points, precision=self.precision)
            simplex_tree = alpha_complex.create_simplex_tree(
                max_alpha_square=self._get_alpha_sq(),
                output_squared_values=output_squared_values
            )

        elif self.filtration_type == "cech":
            
            # Delaunay Cech Complex from a Delaunay triangulation
            if not self.points:
                raise ValueError("DelaunayCechComplex requires points.")
            cech_complex = gudhi.DelaunayCechComplex(points=self.points, precision=self.precision)
            simplex_tree = cech_complex.create_simplex_tree(
                max_alpha_square=self._get_alpha_sq(),
                output_squared_values=output_squared_values
            )

        elif self.filtration_type == "delaunay":
            
            # Delaunay Complex with no filtration
            if not self.points:
                raise ValueError("DelaunayComplex requires points.")
            delaunay_complex = gudhi.DelaunayComplex(points=self.points, precision=self.precision)
            simplex_tree = delaunay_complex.create_simplex_tree(
                filtration=None 
            )

        else:
            raise ValueError("filtration_type must be 'rips', 'alpha', 'cech', or 'delaunay'.")

        self.simplex_tree = simplex_tree  
        
        return simplex_tree

    def _get_alpha_sq(self):

        return self.max_alpha_square if self.max_alpha_square is not None else float('inf')
    
    def get_betti_numbers(self):
        if not hasattr(self, 'simplex_tree'):
            raise RuntimeError("Simplex tree has not been built yet. Call build_simplex_tree() first.")
        
        self.simplex_tree.compute_persistence()
        return self.simplex_tree.betti_numbers()
    
    def plot_persistence_diagram(self, legend=True):
    
        if not hasattr(self, 'simplex_tree'):
            raise RuntimeError("Simplex tree has not been built yet.")
        
        self.simplex_tree.compute_persistence()
    
        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, legend=legend)
        plt.title("Persistence Diagram")
        plt.show()
        
    def plot_persistence_barcode(self):
        
        if not hasattr(self, 'simplex_tree'):
            raise RuntimeError("Simplex tree has not been built yet.")
        
        self.simplex_tree.compute_persistence()
        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_barcode(diag)
        plt.title("Persistence Barcode")
        plt.show()