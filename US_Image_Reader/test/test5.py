import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import rotate

import cupy as cp
import cupyx.scipy.ndimage as cndimage

def gpu_rotate(image, angle, axes):
    img_gpu = cp.array(image)  # Move data to GPU
    rotated_gpu = cndimage.rotate(img_gpu, angle, axes=axes, reshape=False, mode='nearest')
    return cp.asnumpy(rotated_gpu)  # Move data back to CPU for visualization

# Load the NIfTI file
nii_img = nib.load("D:/Ultrasound Data/CAMUS_public/patient0006_4CH_half_sequence.nii")  # Replace with your file path
volume = nii_img.get_fdata()  # Convert to NumPy array

# Get volume dimensions
x_dim, y_dim, z_dim = volume.shape

# Initialize figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)  # Make space for multiple sliders

# Display the middle slice initially
z_index = z_dim // 2
img_display = ax.imshow(volume[:, :, z_index], cmap="gray")
ax.set_title(f"Slice {z_index}/{z_dim}")

# Add sliders for Z-axis (slice selection) and rotation angles
ax_slider_z = plt.axes([0.2, 0.2, 0.65, 0.03])
slider_z = Slider(ax_slider_z, "Slice", 0, z_dim - 1, valinit=z_index, valstep=1)

ax_slider_yaw = plt.axes([0.2, 0.15, 0.65, 0.03])
slider_yaw = Slider(ax_slider_yaw, "Yaw (Z)", -90, 90, valinit=0)

ax_slider_pitch = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_pitch = Slider(ax_slider_pitch, "Pitch (X)", -90, 90, valinit=0)

ax_slider_roll = plt.axes([0.2, 0.05, 0.65, 0.03])
slider_roll = Slider(ax_slider_roll, "Roll (Y)", -90, 90, valinit=0)


# Function to update the displayed slice with rotation
def update(val):
    z = int(slider_z.val)
    yaw_angle = slider_yaw.val
    pitch_angle = slider_pitch.val
    roll_angle = slider_roll.val

    # Extract the selected slice
    slice_2d = volume[:, :, z]
    rotated = volume[:, :, :]
    # Apply rotations
    # rotated = rotate(volume, yaw_angle, axes=(0, 1), reshape=False, mode='nearest')  # Yaw (Z-axis)
    # rotated = rotate(rotated, pitch_angle, axes=(1, 2), reshape=False, mode='nearest')  # Pitch (X-axis)
    # rotated = rotate(rotated, roll_angle, axes=(0, 2), reshape=False, mode='nearest')  # Roll (Y-axis)

    # Update the displayed image
    img_display.set_data(rotated[:, :, z])
    ax.set_title(f"Slice {z}/{z_dim} | Yaw: {yaw_angle:.1f}° | Pitch: {pitch_angle:.1f}° | Roll: {roll_angle:.1f}°")
    fig.canvas.draw_idle()


# Connect sliders to update function
slider_z.on_changed(update)
slider_yaw.on_changed(update)
slider_pitch.on_changed(update)
slider_roll.on_changed(update)

# Show the interactive plot
plt.show()