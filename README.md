This repository contains modular Python tools for 3D medical image analysis and simulation, developed as part of the *Programming Foundations for Medical Image Analysis* module (MPHY0030) at University College London.

### `filter3d.py`
Defines a `Filter3D` class implementing:
- **Gaussian filter** using separable convolution or FFT
- **Bilateral filter** from scratch using NumPy
- Applies filters to volumetric data for anti-aliasing or smoothing

### `image3d.py`
Defines an `Image3D` class for:
- Storing volumetric image data and voxel spacing
- Resizing using `scipy.interpolate.interpn`
- Pre-computed voxel coordinate grids
- Anti-aliased resizing using an input filter
