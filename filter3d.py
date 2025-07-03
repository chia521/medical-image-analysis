#filter3d class script
#Chatgpt was used to assist in understanding the background of coursework.

import numpy as np

class Filter3D:
    ''''
    This class applies 3D filters - Gaussian or Bilateral to volumetric image data.
    '''
    def __init__(self, filter_type, **kwargs):
        """
        Initialize a Filter3D object.
        :param filter_type: str, 'gaussian' or 'bilateral'
        :param kwargs: filter-specific parameters (e.g., sigma for Gaussian)
        """
        self.filter_type = filter_type
        self.params = kwargs

    def apply_filter(self, image3d):
        """
        This function applies the specified filter to the input Image3D object.
        :param image3d: Image3D object
        :return: Filtered 3D NumPy array
        """
        img = image3d.image
        if self.filter_type == 'gaussian':
            return self._apply_gaussian(img, self.params.get('sigma', 1))
        elif self.filter_type == 'bilateral':
            return self._apply_bilateral(img, self.params.get('sigma_spatial', 1), self.params.get('sigma_range', 1))
        else:
            raise ValueError("Unsupported filter type. Use 'gaussian' or 'bilateral'.")

    def _apply_gaussian(self, img, sigma):
        '''
        This function implements Gaussian filter using separable convolutions.
        :param img: 3D NumPy array, the input image.
        :param sigma: Standard deviation for the Gaussian kernel.
        :return: 3D NumPy array, the filtered image.
        '''
        
        tail = int(3 * sigma)
        kernel = np.exp(-0.5 * (np.arange(-tail, tail + 1) ** 2) / sigma ** 2)
        kernel /= kernel.sum()
        filtered = img.copy()
        for axis in range(3):
            filtered = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'),
                axis=axis,
                arr=filtered
            )
        return filtered


    def _apply_bilateral(self, img, sigma_spatial, sigma_intensity):
        """
        Apply a bilateral filter to a 3D image.
        :param img: 3D NumPy array, the input image.
        :param sigma_spatial: Standard deviation for spatial weights.
        :param sigma_intensity: Standard deviation for intensity weights.
        :return: 3D NumPy array, the filtered image.
        """
        # Pad edges to handle boundary conditions
        padded_image = np.pad(img, 1, mode='edge')
        filtered_image = np.zeros_like(img, dtype=np.float64)  # Use float64 for precision

        # Define spatial weights for a 3x3x3 neighborhood
        coords = np.arange(-1, 2)  # [-1, 0, 1]
        grid = np.meshgrid(coords, coords, coords, indexing="ij")
        spatial_weights = np.exp(
            -(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma_spatial ** 2)
        )

        # Iterate over each voxel in the image
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for z in range(img.shape[2]):
                    # Center voxel value
                    center_val = padded_image[x + 1, y + 1, z + 1]

                    # Extract the local neighborhood (3x3x3 patch)
                    patch = padded_image[x:x + 3, y:y + 3, z:z + 3]

                    # Compute intensity weights
                    intensity_weights = np.exp(
                        -np.clip((patch - center_val) ** 2 / (2 * sigma_intensity ** 2), -700, 700)
                    )

                    # Combine spatial and intensity weights
                    weights = spatial_weights * intensity_weights
                    weights_sum = weights.sum()
                    if weights_sum == 0:
                        weights_sum = 1  # Prevent division by zero
                    weights /= weights_sum

                    # Compute the filtered value for the current voxel
                    filtered_image[x, y, z] = np.sum(patch * weights)

        return filtered_image
