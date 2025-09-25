import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import scipy.ndimage as ndimage

# Load the NIfTI file
nii_img = nib.load("D:/Ultrasound Data/CAMUS_public/patient0006_4CH_half_sequence.nii")  # Replace with your file path
volume = nii_img.get_fdata()  # Convert to NumPy array

# Get the volume dimensions
x_dim, y_dim, z_dim = volume.shape
z_index = z_dim // 2  # Start from the middle slice

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

# Display the initial slice
img_display = ax.imshow(volume[:, :, z_index], cmap="gray")
ax.set_title(f"Slice {z_index}/{z_dim}")

# Add a slider for Z-axis slice selection
ax_slider = plt.axes([0.2, 0.2, 0.65, 0.03])
slider = Slider(ax_slider, "Slice", 0, z_dim - 1, valinit=z_index, valstep=1)

# Rotation angles for yaw, roll, and pitch
yaw_angle, roll_angle, pitch_angle = 0, 0, 0  # Initialize angles


# Function to update the displayed slice
def update_slice(val):
    global volume
    z = int(slider.val)
    img_display.set_data(volume[:, :, z])
    ax.set_title(f"Slice {z}/{z_dim}")
    fig.canvas.draw_idle()


slider.on_changed(update_slice)


# Function to apply rotation
def rotate_volume(axis, angle):
    global volume
    if axis == "yaw":  # Rotate around Z-axis
        rotated = ndimage.rotate(volume, angle, axes=(0, 1), reshape=False, order=1)
    elif axis == "roll":  # Rotate around Y-axis
        rotated = ndimage.rotate(volume, angle, axes=(0, 2), reshape=False, order=1)
    elif axis == "pitch":  # Rotate around X-axis
        rotated = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False, order=1)

    volume[:] = rotated  # Update volume
    update_slice(slider.val)  # Refresh display


# Add buttons for Yaw, Roll, and Pitch rotation
ax_button_yaw = plt.axes([0.1, 0.05, 0.2, 0.075])
ax_button_roll = plt.axes([0.4, 0.05, 0.2, 0.075])
ax_button_pitch = plt.axes([0.7, 0.05, 0.2, 0.075])

btn_yaw = Button(ax_button_yaw, "Yaw 15°")
btn_roll = Button(ax_button_roll, "Roll 15°")
btn_pitch = Button(ax_button_pitch, "Pitch 15°")

# Assign functions to buttons
btn_yaw.on_clicked(lambda event: rotate_volume("yaw", 1))
btn_roll.on_clicked(lambda event: rotate_volume("roll", 1))
btn_pitch.on_clicked(lambda event: rotate_volume("pitch", 1))

# Show interactive plot
plt.show()
