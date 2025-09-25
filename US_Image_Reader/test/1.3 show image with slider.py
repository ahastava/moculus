import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the NIfTI file
nii_img = nib.load("D:/Ultrasound Data/CAMUS_public/patient0008_4CH_half_sequence.nii")  # Replace with your file path
volume = nii_img.get_fdata()  # Convert to NumPy array

# Get the volume dimensions
x_dim, y_dim, z_dim = volume.shape

# Initialize figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)

# Display the middle slice initially
z_index = z_dim // 2
img_display = ax.imshow(volume[:, :, z_index], cmap="gray")
ax.set_title(f"Slice {z_index}/{z_dim}")

# Add a slider for Z-axis navigation
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of the slider
slider = Slider(ax_slider, "Slice", 0, z_dim - 1, valinit=z_index, valstep=1)

# Function to update the displayed slice
def update(val):
    z = int(slider.val)
    img_display.set_data(volume[:, :, z])
    ax.set_title(f"Slice {z}/{z_dim}")
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Show the interactive plot
plt.show()