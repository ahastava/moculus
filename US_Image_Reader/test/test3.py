
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# Load the 3D volume (assuming it's a NIfTI file)
nii_img = nib.load("patient0002_2CH_half_sequence.nii")
volume = nii_img.get_fdata()

# Define rotation angles (in degrees)
angle_x = 20  # Rotation around X-axis
angle_y = 0   # Rotation around Y-axis

# Rotate the volume
rotated_volume = ndimage.rotate(volume, angle_x, axes=(1, 2), reshape=False, order=1)

# Extract a middle slice from the rotated volume
slice_index = rotated_volume.shape[2] // 2
slice_view = rotated_volume[:, :, slice_index]

# Display the angled view
plt.imshow(slice_view, cmap="gray")
plt.title(f"Angled View (Rotated {angle_x}° Around X-axis)")
plt.axis("off")
plt.show()