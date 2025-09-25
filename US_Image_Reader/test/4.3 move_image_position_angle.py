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
plt.subplots_adjust(bottom=0.4)  # Make space for multiple sliders

# Display the middle slice initially
z_index = z_dim // 2
img_display = ax.imshow(volume[:, :, z_index], cmap="gray")
ax.set_title(f"Slice {z_index}/{z_dim}")

# Add sliders for Z-axis (slice selection), rotation angles, and translation
ax_slider_z = plt.axes([0.2, 0.3, 0.65, 0.03])
slider_z = Slider(ax_slider_z, "Slice", 0, z_dim - 1, valinit=z_index, valstep=1)

ax_slider_yaw = plt.axes([0.2, 0.25, 0.65, 0.03])
slider_yaw = Slider(ax_slider_yaw, "Yaw (Z)", -90, 90, valinit=0)

ax_slider_pitch = plt.axes([0.2, 0.2, 0.65, 0.03])
slider_pitch = Slider(ax_slider_pitch, "Pitch (X)", -90, 90, valinit=0)

ax_slider_roll = plt.axes([0.2, 0.15, 0.65, 0.03])
slider_roll = Slider(ax_slider_roll, "Roll (Y)", -90, 90, valinit=0)

ax_slider_x = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_x = Slider(ax_slider_x, "Move X", -100, 100, valinit=0)

ax_slider_y = plt.axes([0.2, 0.05, 0.65, 0.03])
slider_y = Slider(ax_slider_y, "Move Y", -100, 100, valinit=0)

# Apply translation with zero padding instead of wrapping
def shift_image(image, shift_x, shift_y):
    shifted = np.zeros_like(image)  # Create an empty black canvas of the same shape

    # Determine the valid range to copy pixels
    x_start = max(0, shift_x)
    x_end = min(image.shape[1], image.shape[1] + shift_x)
    y_start = max(0, shift_y)
    y_end = min(image.shape[0], image.shape[0] + shift_y)

    x_start_src = max(0, -shift_x)
    x_end_src = min(image.shape[1], image.shape[1] - shift_x)
    y_start_src = max(0, -shift_y)
    y_end_src = min(image.shape[0], image.shape[0] - shift_y)

    # Copy valid pixels while keeping empty regions black
    shifted[y_start:y_end, x_start:x_end] = image[y_start_src:y_end_src, x_start_src:x_end_src]

    return shifted




# Function to update the displayed slice with rotation and translation
def update(val):
    z = int(slider_z.val)
    yaw_angle = slider_yaw.val
    pitch_angle = slider_pitch.val
    roll_angle = slider_roll.val
    shift_x = int(slider_x.val)
    shift_y = int(slider_y.val)

    # Extract the selected slice
    slice_2d = volume[:, :, z]

    # Apply rotations (commented GPU implementation for now)
    rotated = rotate(slice_2d, yaw_angle, axes=(0, 1), reshape=False, mode='nearest')  # Yaw (Z-axis)
    rotated = rotate(rotated, pitch_angle, axes=(1, 0), reshape=False, mode='nearest')  # Pitch (X-axis)
    rotated = rotate(rotated, roll_angle, axes=(0, 1), reshape=False, mode='nearest')  # Roll (Y-axis)

    # Apply translation using np.roll
   # moved = np.roll(rotated, shift_x, axis=1)  # Shift along X-axis
   # moved = np.roll(moved, shift_y, axis=0)  # Shift along Y-axis

    # Apply translation using zero-padding instead of wrapping
    moved = shift_image(rotated, shift_x, shift_y)

    # Update the displayed image
    img_display.set_data(moved)
    ax.set_title(f"Slice {z}/{z_dim} | Yaw: {yaw_angle:.1f}° | Pitch: {pitch_angle:.1f}° | Roll: {roll_angle:.1f}° | Move X: {shift_x} | Move Y: {shift_y}")
    fig.canvas.draw_idle()


# Connect sliders to update function
slider_z.on_changed(update)
slider_yaw.on_changed(update)
slider_pitch.on_changed(update)
slider_roll.on_changed(update)
slider_x.on_changed(update)
slider_y.on_changed(update)

# Show the interactive plot
plt.show()
