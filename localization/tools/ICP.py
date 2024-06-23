import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

def calculate_centroid(points: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid of a set of points.

    Parameters:
        points (np.ndarray): An array of points of shape (n_points, 3).

    Returns:
        np.ndarray: The centroid of the points.
    """
    return np.mean(points, axis=0)

def rigid_transformation(cloud: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Applies a rigid transformation to a point cloud.
    
    Parameters:
        cloud (np.ndarray): A point cloud array of shape (n_points, 3).
        R (np.ndarray): Rotation matrix of shape (3, 3).
        t (np.ndarray): Translation vector of shape (3, 1).
    
    Returns:
        np.ndarray: Transformed point cloud of shape (n_points, 3).
    """
    # Validate input types
    if not isinstance(cloud, np.ndarray):
        raise TypeError("The point cloud must be a numpy array.")
    if not isinstance(R, np.ndarray):
        raise TypeError("The rotation matrix must be a numpy array.")
    if not isinstance(t, np.ndarray):
        raise TypeError("The translation vector must be a numpy array.")

    # Validate input dimensions
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("Point cloud shape must be (n_points, 3).")
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    if t.shape != (3, 1):
        raise ValueError("Translation vector must have shape (3, 1).")

    # Validate matrix compatibility
    if cloud.shape[1] != R.shape[0]:
        raise ValueError("The number of columns in the point cloud must match the number of rows in the rotation matrix.")
    
    return (R @ cloud.T + t).T

class ICP:
    """
    Implementation of an ICP class
    """

    def __init__(
        self, max_iterations: int = 50, max_dist: float = 10, tol: float = 1e-5
    ) -> None:
        """
        Args:
            max_iterations (int, optional): Maximum iterations for the ICP algorithm. Defaults to 50.
            max_dist (float, optional): Max distance to cosider coincidences. Defaults to 10.
            tol (float, optional): Maximum tolerance to check if the algorithm converged. Defaults to 1e-5.
        """
        self.max_iterations = max_iterations
        self.max_dist = max_dist
        self.tol = tol

    def align(
        self, source: np.array, target: np.array):
        """
        Align function to compute the transformation from source to target using Single Value Decomposition.

        Args:
            source (np.array): Pointcloud to be aligned
            target (np.array): Base Pointcloud, in localization, this is the map

        Returns:
            Tuple[np.array, np.array]: Returns the rotation matrix and translation from source to target.
            Also returns a list with all Rotations and translations for each iteration in ICP.

        """
        
        # Calculate centroids of the source and target points
        centroid_target = calculate_centroid(target)

        # Center the points around the centroid
        centered_target = target - centroid_target
        
        # Build KDTree
        tree = KDTree(target)

        # Initilialize R and t
        R = np.eye(target.shape[1])
        t = np.zeros((target.shape[1], 1))

        R_list = [R]
        t_list = [t]
        corres_values_list = []

        for iteration in range(self.max_iterations):

            # Find nearest neighbors
            distances, indices = tree.query(source)
            # Compute correspondences
            correspondences = np.asarray([(i, j) for i, j in enumerate(indices)])
            mask = distances < self.max_dist
            
            # Filter correspondences
            correspondences = correspondences[mask, :]
            
            # Sort points according to correspondence
            sorted_source = source[correspondences[:, 0]]
            sorted_target = target[correspondences[:, 1]]

            # Get centroid of sorted points
            centroid_source = calculate_centroid(sorted_source)
            centroid_target = calculate_centroid(sorted_target)
            
            # Center sorted points
            centered_source = sorted_source - centroid_source
            centered_target = sorted_target - centroid_target
            
            #  Compute covariance matrix
            S = centered_source.T @ centered_target
            # Perform Singular Value Decomposition
            U, _, V = np.linalg.svd(S)
            
            # Compute the rotation matrix, ensuring a right-handed coordinate system
            Rn = V.T @ U.T

            # Compute the translation vector
            tn = centroid_target - Rn @ centroid_source
            tn = tn[:,np.newaxis]
            source = rigid_transformation(source, Rn, tn)
            
            # Update transformation
            t = Rn @ t + tn
            R = np.dot(R, Rn)

            t_list.append(t.copy())
            R_list.append(R.copy())
            corres_values_list.append(correspondences.copy())

            if np.allclose(tn, 0, atol=self.tol) and np.allclose(Rn, np.eye(Rn.shape[0]), atol=self.tol):
                break

        return R, t, R_list, t_list, corres_values_list, source