import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file

import os


filename = "D:/Ultrasound Data/CAMUS_public/patient0008_4CH_half_sequence.nii"

nii_img = nib.load(filename)

# Get image data as a NumPy array
img_data = nii_img.get_fdata()
print(nii_img.shape)



# Display a middle slice (assuming a 3D volume)
slice_index = img_data.shape[2] // 2  # Adjust for desired axis
plt.imshow(img_data[:, :, 0], cmap="gray")
plt.title("Middle Slice of NIfTI Image")
plt.axis("off")
plt.show()