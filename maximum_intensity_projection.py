#class file for maximum intensity projection
#Chatgpt was used to assist in understanding the background of coursework.

import numpy as np
from scipy.interpolate import interpn

class MaximumIntensityProjection:
    """
    Class that generate a 2D digitally reconstructed radiography (DRR) image at specified detector position
    """
    def __init__(self, source_to_detector_dist, detector_size, drr_voxdims):
        """
        Initialize the MaximumIntensityProjection class.

        Parameters:
        - source_to_detector_dist (float): Distance from the source to the detector.
        - detector_size (tuple of int): Dimensions of the detector (width, height).
        - drr_voxdims (tuple of float): Voxel dimensions of the detector in mm.
        """
        self.source_to_detector_dist = source_to_detector_dist 
        self.detector_size = detector_size 
        self.drr_voxdims = drr_voxdims 

    def project(self, volume, vol_voxdims, image_position=(0, 0, 0)):
        """
        Generate the 2D DRR image by projecting a 3D volume onto a 2D detector.

        Parameters:
        - volume (numpy.ndarray): 3D volume to be projected.
        - vol_voxdims (tuple of float): Voxel dimensions of the 3D volume (dx, dy, dz) in mm.
        - image_position (tuple of float): Position of the image relative to the detector, defined as:
            - image_position[0]: X-axis offset (left-right, in mm).
            - image_position[1]: Y-axis offset (up-down, in mm).
            - image_position[2]: Z-axis offset (source-to-detector distance, in mm).

        Returns:
        - numpy.ndarray: 2D DRR image (maximum intensity projection).
        """
        # Define coordinates for the 3D volume centered at its origin
        vol_i = np.linspace(-volume.shape[0] / 2 + 0.5, volume.shape[0] / 2 - 0.5, volume.shape[0]) * vol_voxdims[0]
        vol_j = self.source_to_detector_dist - np.linspace(volume.shape[1] - 0.5, 0.5, volume.shape[1]) * vol_voxdims[1]
        vol_k = np.linspace(-volume.shape[2] / 2 + 0.5, volume.shape[2] / 2 - 0.5, volume.shape[2]) * vol_voxdims[2]

        # Create the 2D detector grid centered at its origin
        drr_i, drr_j = np.meshgrid(
            np.linspace(-self.detector_size[0] / 2 + 0.5, self.detector_size[0] / 2 - 0.5, self.detector_size[0]) * self.drr_voxdims[0],
            np.linspace(-self.detector_size[1] / 2 + 0.5, self.detector_size[1] / 2 - 0.5, self.detector_size[1]) * self.drr_voxdims[1],
            indexing='ij'
        )
    
        # Apply the image position offsets to the detector grid
        drr_i += image_position[0]  # X-axis offset
        drr_j += image_position[1]  # Y-axis offset
        self.source_to_detector_dist += image_position[2]  # Z-axis offset

        # Operations to obtain range of radial distance,r:
        # Compute the distance from the source to each voxel in the volume
        vol_ds = np.sqrt(sum([x**2 for x in np.meshgrid(vol_i,vol_j,vol_k,indexing='ij')]))
         # Compute the distance from the source to each point on the detector
        drr_ds = np.sqrt(drr_i**2 + drr_j**2 + self.source_to_detector_dist**2)
        # Calculate the maximum and minimum radial distances for sampling
        r_max = max(vol_ds.max(), drr_ds.max())
        r_min = min(vol_ds.min(), drr_ds.min())
        # Estimate the number of samples along the projection ray
        n_samples = int(np.ceil(1.5 * (r_max - r_min)))

        # Angles required for Spherical to Cartesian conversion (azimuth and elevation angles)
        az = np.arctan2(drr_i, drr_j)[..., np.newaxis]
        el = np.arctan2(self.source_to_detector_dist, np.sqrt(drr_i**2 + drr_j**2))[..., np.newaxis]

        # Generate r for sampling
        r = np.reshape(np.linspace(r_min,r_max,n_samples),(1,1,n_samples))
        
        # Convert Spherical coordinates to Cartesian coordinates
        sample_z = r * np.cos(el) * np.cos(az)
        sample_y = r * np.cos(el) * np.sin(az)
        sample_x = r * np.sin(el)

        # Interpolate intensity values from the 3D volume along the ray
        samples = interpn(
                        (vol_i, vol_j, vol_k),
                        volume,
                        np.stack([sample_y, sample_x, sample_z], axis=3),
                        method='linear',
                        bounds_error=False,
                        fill_value=0.0
                        )

        # Generate 2D DRR by selecting the maximum intensity along the ray
        return np.amax(samples, axis=2)
    




