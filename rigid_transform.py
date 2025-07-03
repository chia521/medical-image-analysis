#class file for rigid transformation
#Chatgpt was used to assist in understanding the background of coursework.

import numpy as np
from scipy.interpolate import interpn

class RigidTransform:
    """
    Class that perform 3D rigid transformations on image volumes.
    - Computes a dense displacement field (DDF) to represent transformations.
    - Applies rigid transformations to warp 3D image volumes.
    """
    def __init__(self, rotations, translations, warped_image_size, voxdims):
        """
        Initialize the RigidTransform class with rotation angles, translation distances, 
        and voxel dimensions. Precomputes the rotation matrix, translation vector, 
        and dense displacement field for efficient transformation.

        Parameters:
        - rotations (tuple of float): Rotation angles (rx, ry, rz) in radians for the x, y, and z axes.
        - translations (tuple of float): Translation distances (tx, ty, tz) in mm for x, y, and z axes.
        - warped_image_size (tuple of int): Dimensions of the transformed image (nx, ny, nz).
        - voxdims (tuple of float): Voxel sizes (dx, dy, dz) in mm.
        """
        self.rotations = rotations
        self.translations = translations
        self.warped_image_size = warped_image_size
        self.voxdims = voxdims

        # Precompute the rotation matrix
        self.rotation_matrix = self._compute_rotation_matrix(*rotations)

        # Precompute the translation vector
        self.translation_vector = np.array(translations)

        # Precompute the Dense Displacement Field (DDF)
        self.ddf = self._compute_ddf()

    def _compute_rotation_matrix(self, rx, ry, rz):
        """
        Compute the 3D rotation matrix from the given rotation angles.

        Parameters:
        - rx, ry, rz: Rotation angles for x, y, and z axes (in radians).

        Returns:
        - numpy.ndarray: 3x3 rotation matrix.
        """
        # Rotation about x-axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        # Rotation about y-axis
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        # Rotation about y-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        # Combine the rotations
        return Rz @ Ry @ Rx  

    def _compute_ddf(self):
        """
        Compute the dense displacement field (DDF) for the rigid transformation.

        Returns:
        - numpy.ndarray: Dense displacement field of shape (nx, ny, nz, 3).

        Coordinate system:
          - Origin: Center of the image.
          - Orientation:
                - X-axis (left-right, in mm).
                - Y-axis (anterior-posterior, in mm).
                - Z-axis (superior-inferior/up-down, in mm).
        """
        # Define the size and spacing of the grid
        nx, ny, nz = self.warped_image_size
        dx, dy, dz = self.voxdims

        # Create a coordinate system centered at the image
        x = np.linspace(-nx / 2 + 0.5, nx / 2 - 0.5, nx) * dx
        y = np.linspace(-ny / 2 + 0.5, ny / 2 - 0.5, ny) * dy
        z = np.linspace(-nz / 2 + 0.5, nz / 2 - 0.5, nz) * dz

        # Create a meshgrid for the warped voxel coordinates
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1)

        # Apply rotation and translation
        rotated_points = np.dot(grid_points, self.rotation_matrix.T)
        transformed_points = rotated_points + self.translation_vector

        # Compute the DDF: difference between the original and transformed coordinates
        ddf = grid_points - transformed_points
        return ddf

    def warp(self, volume):
        """
        Apply rigid transformation to the given 3D volume using precomputed DDF.

        Parameters:
        - volume (numpy.ndarray): The 3D volume to be warped.

        Returns:
        - numpy.ndarray: The warped 3D volume.
        """
        # Defining grid coordinates for input volume
        nx, ny, nz = volume.shape
        dx, dy, dz = self.voxdims

        # Create the grid for the input volume
        x = np.linspace(-nx / 2 + 0.5, nx / 2 - 0.5, nx) * dx
        y = np.linspace(-ny / 2 + 0.5, ny / 2 - 0.5, ny) * dy
        z = np.linspace(-nz / 2 + 0.5, nz / 2 - 0.5, nz) * dz

        # Apply DDF to compute the original coordinates
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        warped_points = np.stack([grid_x, grid_y, grid_z], axis=-1)
        original_points = warped_points + self.ddf

        # Interpolate intensity values from the original image to the warped coordinates
        warped_volume = interpn(
            (x, y, z),
            volume,
            original_points.reshape(-1, 3),
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Reshape the warped image back to the original dimensions
        return warped_volume.reshape(volume.shape)




