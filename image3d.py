#image3d class script
#Chatgpt was used to assist in understanding the background of coursework.


import numpy as np
from scipy.interpolate import interpn

class Image3D:
    '''
    This class handles 3D medical images with different voxel dimension, image sizes and data types.
    '''
    def __init__(self, image, voxel_dim):
        """
        Initialize an Image3D object.
        :param image: 3D NumPy array
        :param voxel_dim: Tuple of three floats, voxel dimensions
        """
        self.image = image
        self.voxel_dim = voxel_dim
        self.coords = tuple(np.arange(dim) * self.voxel_dim[i] for i, dim in enumerate(self.image.shape))

    def _compute_voxel_coordinates(self):
        # Compute voxel coordinates
        shape = self.image.shape
        return tuple(np.arange(dim) * self.voxel_dim[i] for i, dim in enumerate(shape))

    def volume_resize(self, resize_ratio):
        """
        Resize the volume using the specified resize ratio.
        :param resize_ratio: Tuple of three floats
        :return: Resized Image3D object
        """
        output_shape = [int(dim * ratio) for dim, ratio in zip(self.image.shape, resize_ratio)]
        new_voxel_dim = [dim / size for dim, size in zip(self.voxel_dim, output_shape)]
        new_coords = tuple(
            np.linspace(0, (dim - 1) * voxel, size)
            for dim, voxel, size in zip(self.image.shape, self.voxel_dim, output_shape)
        )
        grid = np.meshgrid(*new_coords, indexing="ij")
        new_grid = np.array(grid).reshape(3, -1).T

        # Interpolate using interpn
        resized_image = interpn(
            self.coords,  # Tuple of 1D arrays for the original grid
            self.image,    # Original 3D data
            new_grid,     # Flattened 2D array of target coordinates
            method='linear',
            bounds_error=False,
            fill_value=0
        ).reshape(output_shape)

        return Image3D(resized_image, new_voxel_dim)

    def volume_resize_antialias(self, resize_ratio, filter3d):
        """
        Resize the volume after applying an anti-aliasing filter.
        :param resize_ratio: Tuple of three floats
        :param filter3d: Filter3D object
        :return: Resized Image3D object
        """
        filtered_image = filter3d.apply_filter(self)
        filtered_image3d = Image3D(filtered_image, self.voxel_dim)
        return filtered_image3d.volume_resize(resize_ratio)
