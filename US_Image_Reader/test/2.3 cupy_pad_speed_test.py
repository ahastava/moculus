import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import rotate

from datetime import datetime

import cupy as cp
import cupyx.scipy.ndimage as cndimage
from scipy.ndimage import zoom

def gpu_rotate(image, angle, axes):
    img_gpu = cp.array(image)  # Move data to GPU
    rotated_gpu = cndimage.rotate(img_gpu, angle, axes=axes, reshape=False, mode='nearest')
    return cp.asnumpy(rotated_gpu)  # Move data back to CPU for visualization

# Load the NIfTI file
nii_img = nib.load("D:/Ultrasound Data/CAMUS_public/patient0006_4CH_half_sequence.nii")  # Replace with your file path
volume = nii_img.get_fdata()  # Convert to NumPy array

import time

start_time = time.time()

scale_factor = 0.2  # Scale down by 1/2
resized_volume = zoom(volume, zoom=(scale_factor, scale_factor, scale_factor),
                      order=3)  # order=3 for cubic interpolation
end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU scale Elapsed time: {elapsed_time:.3f} seconds")



start_time = time.time()
max_dim = max(volume.shape)

# #pad_width = [(s // 2, s // 2) for s in self.volume.shape]
pad_width = [((max_dim - s) // 2, (max_dim - s + 1) // 2) for s in volume.shape]  # Ensure symmetry

# # Apply padding with zeros
padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU pad Elapsed time: {elapsed_time:.3f} seconds")



start_time = time.time()
volume_gpu = cp.array(volume)  # Move data to GPU

end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU to GPU Elapsed time: {elapsed_time:.3f} seconds")

# ------------------------------------
start_time = time.time()
# Resize the volume by a factor of 0.2
scale_factor = 0.2
resized_gpu = cndimage.zoom(volume_gpu, zoom=scale_factor, order=1)  # order=1 for bilinear interpolation

end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU scale Elapsed time: {elapsed_time:.3f} seconds")

# ------------------------------------


start_time = time.time()

# Find the maximum size among all dimensions
max_dim = max(volume.shape)

# Calculate padding for each axis to match max_dim
pad_width = [((max_dim - s) // 2, (max_dim - s + 1) // 2) for s in volume.shape]

# Convert to CuPy tuple format
pad_width_gpu = tuple((cp.int32(p[0]), cp.int32(p[1])) for p in pad_width)

# Apply padding on GPU
padded_volume_gpu = cp.pad(volume_gpu, pad_width_gpu, mode='constant', constant_values=0)


end_time = time.time()

# ------------------------------------

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time
print(f"GPU pad Elapsed time: {elapsed_time:.3f} seconds")

start_time = time.time()
# Move the padded volume back to CPU
padded_volume = cp.asnumpy(padded_volume_gpu)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU to CPU Elapsed time: {elapsed_time:.3f} seconds")