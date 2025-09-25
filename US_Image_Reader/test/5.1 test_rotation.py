import numpy as np
from scipy.ndimage import rotate


# Define the 3D matrix
matrix = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])

matrix = np.array([
    [
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
],
    [
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
],
    [
    [0,0, 0,0, 0],
    [0,0, 0,20,0],
    [0,0,10,30,0],
    [0,0, 0,20,0],
    [0,0, 0,0,0]
],
    [
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
],
    [
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
]
])

# Rotate the matrix by 90 degrees around the X-axis
#rotated_matrix = rotate(matrix, 90, axes=(1, 2), reshape=False, mode='nearest')
rotated_matrix = rotate(matrix, 45, axes=(2, 0), reshape=False, mode='nearest')
#rotated_matrix = rotate(matrix, 180, axes=(2, 0), reshape=False, mode='nearest')

# Print the original and rotated matrix
print("Original Matrix:")
print(matrix)

print("\nRotated Matrix (by 90 degrees around X-axis):")
print(rotated_matrix)

# # Optional: Display the original and rotated matrices using Matplotlib
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
# # Display the original matrix slice (0th slice in Z-axis)
# ax[0].imshow(matrix[0], cmap='gray')
# ax[0].set_title("Original Matrix (Slice 0)")
# ax[0].axis('off')
#
# # Display the rotated matrix slice (0th slice in Z-axis)
# ax[1].imshow(rotated_matrix[0], cmap='gray')
# ax[1].set_title("Rotated Matrix (Slice 0)")
# ax[1].axis('off')
#
# plt.tight_layout()
# plt.show()