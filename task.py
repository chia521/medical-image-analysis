#task

import matplotlib.pyplot as plt
from scipy.io import loadmat
from rigid_transform import RigidTransform
from maximum_intensity_projection import MaximumIntensityProjection
import numpy as np

# Load data
data = loadmat('/home/ccl521/MPHY0030/cw1.notfinal/task2.notfinal/test_lung_ct.mat')
volume = data['vol'].astype('float16')
voxel_dims = data['voxdims'][0] #giving [1 1 1]


# Experiment 1: Project image volume by Rigid Transformations 
# Define transformations for various rotation and translation 
transformations = {
    "a1": ([0, 0, 0], [40, 0, 0]),  # a1: Translation in x only
    "a2": ([0, 0, 0], [0, 50, 0]),  # a2: Translation in y only 
    "a3": ([0, 0, 0], [0, 0, 40]),  # a3: Translation in z only
    "b1": ([np.pi/6, 0, 0], [0, 0, 0]),  # b1: Rotation about x only
    "b2": ([0, np.pi/6, 0], [0, 0, 0]),  # b2: Rotation about y only
    "b3": ([0, 0, np.pi/6], [0, 0, 0]),  # b3: Rotation about z only

    "c1": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-100,100), 
            np.random.uniform(-40, 40)] ),  #c1: Random sampling

    "c2": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-100,100), 
            np.random.uniform(-40, 40)] ),  #c2: Random sampling

    "c3": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-100,100), 
            np.random.uniform(-40, 40)] )  #c3: Random sampling
}


# Loop through transformations
for key, (rotations, translations) in transformations.items():
    # Apply rigid transformation (rotation and translation)
    rt = RigidTransform(rotations, translations, volume.shape, voxel_dims)
    transformed_volume = rt.warp(volume)

    # Apply MIP to generate DRR
    mip = MaximumIntensityProjection(1000, (200, 120), (0.8, 0.8))   
    drr = mip.project(transformed_volume, voxel_dims, (0, 0, 0))  #image position is fixed at (0,0,0)

    # Save DRR as PNG
    plt.imshow(drr, cmap='gray')
    plt.axis('off')
    plt.savefig(f'exp1_{key}.png')

print("Experiment 1: DRRs saved as exp1_a1.png to exp1_c3.png.")


# Experiment 2: Project image volume by re-positioning image volume
# Define transformations on rotations and detector repositioning
transformations = {
    "a1": ([0, 0, 0], [50, 0, 0]),  # a1: Reposition in x only
    "a2": ([0, 0, 0], [0, 40, 0]),  # a2: Reposition in y only
    "a3": ([0, 0, 0], [0, 0, 500]),  # a3: Reposition in z only
    "b1": ([np.pi / 6, 0, 0], [0, 0, 0]),  # b1: Rotation about x only
    "b2": ([0, np.pi / 6, 0], [0, 0, 0]),  # b2: Rotation about y only
    "b3": ([0, 0, np.pi / 6], [0, 0, 0]),  # b3: Rotation about z only

    "c1": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-40, 40), 
            np.random.uniform(-400, 400)] ),  #c1: Random sampling

    "c2": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-40, 40), 
            np.random.uniform(-400, 400)] ),  #c1: Random sampling

    "c3": ([np.random.uniform(-np.pi / 6, np.pi / 6) for _ in range(3)],
           [np.random.uniform(-40, 40), 
            np.random.uniform(-40, 40), 
            np.random.uniform(-400, 400)] )  #c1: Random sampling
 }



# Loop through transformations
for key, (rotations, image_position) in transformations.items():
    # Apply rigid transformation (only rotations are applied, no translation)
    rt = RigidTransform(rotations, (0, 0, 0), volume.shape, voxel_dims)  #translation is fixed at (0, 0, 0)
    transformed_volume = rt.warp(volume)

    # Apply MIP projection with the specified `image_position` to generate DRR
    mip = MaximumIntensityProjection(1000, (200, 120), (0.8, 0.8))
    drr = mip.project(transformed_volume, voxel_dims, image_position)

    # Save DRR as PNG
    plt.imshow(drr, cmap='gray')
    plt.axis('off')
    plt.savefig(f"exp2_{key}.png")

print("Experiment 2: DRRs saved as exp1_a1.png to exp1_c3.png.")


#Comparing two sets of DDRs
#For Exp1, when the volume traslates along y-axis (a2), different levels 
#of anatomy along that axis are revealed. As translation in y-axis increases, the anatomy 
#appears to reveal deeper structures. However, the size of the anatomy remains the same because 
#the projection setup does not change. For Exp2, when the volume is repositioned neareror farther 
#from the detector (a3), its apparent size changes. As it moves closer to the detector, the size 
#appears bigger due to changing scale. This explains why DRRs in Exp1 appear the same size, while 
#in Exp2, the anatomy’s size varies. 


