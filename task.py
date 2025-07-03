#task script

import numpy as np
from image3d import Image3D
from filter3d import Filter3D
from PIL import Image
import time

# Load image
image = np.load("image_train00.npy")
voxel_dim = (2.0, 0.5, 0.5)  # Example voxel dimensions
img3d = Image3D(image, voxel_dim)


# Experiment 1: Volume resizing
print("\nExperiment 1: Volume resizing")
# Calculate resizing ratio
resize_ratios = {
    "scenario1": (1.5, 1.5, 1.5), #upsampling
    "scenario2": (0.5, 0.5, 0.5), #downsampling
    "scenario3": (1.0, 4.0, 4.0), #resampling to isotropic voxel dimension
}

# Define filtering options
filters = {
    "a": None,  #no filter
    "b": Filter3D("gaussian", sigma=1),  #gaussian
    "c": Filter3D("bilateral", sigma_spatial=2, sigma_range=0.1)  #bilateral
}

# Iterate over each resizing scenario
for scenario, ratio in resize_ratios.items():
    for filter_name, filter3d in filters.items():
        start_time = time.time()

        # Apply resizing with or without filtering
        #for no filter, volume_resize is applied to image; for gaussian and bilateral filters, volume_resize_antialias is applied.
        if filter3d:
            resized_img = img3d.volume_resize_antialias(ratio, filter3d) 
        else:
            resized_img = img3d.volume_resize(ratio)
        

        elapsed = time.time() - start_time
        print(f"{scenario} with {filter_name}: {elapsed:.2f}s")

         # Select 5 evenly spaced slices along the axial (z) direction
        slice_indices = np.linspace(0, resized_img.image.shape[0] - 1, 5, dtype=int)

        # Normalize the resized image to range [0, 255] for saving as PNG
        scaled_image = ((resized_img.image - np.min(resized_img.image)) /
                (np.max(resized_img.image) - np.min(resized_img.image)) * 255).astype(np.uint8)

        # Save axial slices
        for z in slice_indices:
            slice_img = Image.fromarray(scaled_image[z, :, :].astype("uint8"))  # Axial slice
            slice_img.save(f"exp1_{scenario}_z{z}_{filter_name}.png")



#Resizing without filter takes the shortest time, followed by with Gaussian filter and Bilateral filter.
#Downsampling takes the shortest time, followed by up sampling and resampling to isotropic voxel.
#Resizing without filter is computationally efficient but many cause aliasing; Gaussian filtering reduces 
#aliasing effect through smoothing; Bilateral filter is computationally expensive but is able to provide 
#smoothing while preserving edge.



# Experiment 2: Investigate aliasing effects
print("\nExperiment 2: Investigating aliasing effects")

#Load image
image = np.load("image_train00.npy")
voxel_dim = (2.0, 0.5, 0.5)  # Example voxel dimensions
img3d = Image3D(image, voxel_dim)

# Define filters
filters = {
    "a": None,  # No filter
    "b": Filter3D("gaussian", sigma=1),  # Gaussian filter
    "c": Filter3D("bilateral", sigma_spatial=5, sigma_range=100)  # Bilateral filter
}

# Up-sample the original image (Scenario 1 from Experiment 1)
upsample_ratio = (1.5, 1.5, 1.5)
upsampled_image = img3d.volume_resize_antialias(upsample_ratio, Filter3D("bilateral", sigma_spatial=2, sigma_range=0.1))  ## Bilateral filter during upsampling

# Down-sample back to the original size
original_size_ratio = tuple(1 / r for r in upsample_ratio)

# Then apply each filter
for filter_name, filter3d in filters.items():  
        start_time = time.time()

        # Down-sample using the strategy
        if filter3d:
            downsampled_image = upsampled_image.volume_resize_antialias(original_size_ratio, filter3d)
        else:
            downsampled_image = upsampled_image.volume_resize(original_size_ratio)

        elapsed = time.time() - start_time
        print(f"Down-sampling with {filter_name}: {elapsed:.2f}s")

     
        # Compute intensity differences
        intensity_diff = -(img3d.image - downsampled_image.image)
        mean_diff = np.mean(intensity_diff)
        std_diff = np.std(intensity_diff)
        print(f"Filter: {filter_name} - Mean intensity difference: {mean_diff:.2f}, Std deviation: {std_diff:.2f}")


        # Ensure normalization of the image to 0-255 range
        #slice_array = downsampled_image[z, :, :]
        slice_indices = np.linspace(0, downsampled_image.image.shape[0] - 1, 5, dtype=int)
        scaled_image = ((downsampled_image.image - np.min(downsampled_image.image)) /
                (np.max(downsampled_image.image) - np.min(downsampled_image.image)) * 255).astype(np.uint8)
        
        for z in slice_indices: 
            # Save the rotated slice as an image with the correct ordering
            slice_img = Image.fromarray(scaled_image[z, :, :].astype("uint8"))
            slice_img.save(f"exp2_scenario2_z{z}_{filter_name}.png")


#No filter shows highest mean intensity difference which is expected as aliasing is introduced.
#Bilateral filtering gives slightly lower mean inensity difference and 
# std deviation compared to no filtering. Gaussian filtering gives lowest mean intensity 
# difference and std deviation. This is unusual as Bilateral is supposed to have the lowest 
#mean intensity difference as it minimizes aliasing while preserving edge. It is assumed
#that the resizing ratios and filter parameters are not finely tuned yet.