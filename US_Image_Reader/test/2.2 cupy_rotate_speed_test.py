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

start_time = time.time()  # Start timing


yaw_angle = 1

rotated = rotate(volume, yaw_angle, axes=(0, 1), reshape=False, mode='nearest')  # Yaw (Z-axis)

# End timing
end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time
print(f"CPU Rotation Elapsed time: {elapsed_time:.3f} seconds")

start_time = time.time()
img_gpu = cp.array(volume)  # Move data to GPU

end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time
print(f"CPU to GPU Elapsed time: {elapsed_time:.3f} seconds")

for i in range(5):
    start_time = time.time()
    rotated_gpu = cndimage.rotate(img_gpu, yaw_angle, axes=(0, 1), reshape=False, mode='nearest')
    end_time = time.time()

    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time
    print(f"GPU Rotation Elapsed time {i}: {elapsed_time:.3f} seconds")


start_time = time.time()
a = cp.asnumpy(rotated_gpu)  # Move data back to CPU for visualization

# End timing
end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time
print(f"GPU to CPU Elapsed time: {elapsed_time:.3f} seconds")
