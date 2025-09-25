import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file

import os

folder_path = "D:/Ultrasound Data/CAMUS_public/"  # Change to your directory
filenames = os.listdir(folder_path)

#filename = "D:/Ultrasound Data/CAMUS_public/patient0002_2CH_half_sequence.nii"

for filename in filenames:

    print(filename)


#(748, 584, 17)


    nii_img = nib.load(folder_path+filename)

    # Get image data as a NumPy array
    img_data = nii_img.get_fdata()
    print(nii_img.shape)


# patient0002_2CH_half_sequence.nii
# (748, 584, 17)
# patient0004_2CH_half_sequence.nii
# (641, 454, 19)
# patient0004_4CH_half_sequence.nii
# (641, 454, 18)
# patient0006_4CH_half_sequence.nii
# (669, 552, 19)
# patient0008_4CH_half_sequence.nii
# (787, 649, 27)
# patient0010_4CH_half_sequence.nii
# (669, 552, 26)
# patient0012_4CH_half_sequence.nii
# (512, 422, 22)
# patient0014_4CH_half_sequence.nii
# (590, 487, 22)



# Display a middle slice (assuming a 3D volume)
# slice_index = img_data.shape[2] // 2  # Adjust for desired axis
# plt.imshow(img_data[:, :, slice_index], cmap="gray")
# plt.title("Middle Slice of NIfTI Image")
# plt.axis("off")
# plt.show()