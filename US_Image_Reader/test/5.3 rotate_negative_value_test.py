import numpy as np
from scipy.ndimage import rotate

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
rotated_matrix = rotate(matrix, 135, axes=(2, 0), reshape=False, mode='constant', cval=0.0, order=2)
#rotated_matrix = rotate(matrix, 180, axes=(2, 0), reshape=False, mode='nearest')

# Print the original and rotated matrix
print("Original Matrix:")
print(matrix)

print("\nRotated Matrix (by 90 degrees around X-axis):")
print(rotated_matrix)

rotated_matrix = np.clip(rotated_matrix, 0, None)

# Get first nonzero index along Z-axis
first_nonzero_idx = np.argmax(rotated_matrix != 0, axis=2)  # Indices where first nonzero appears

# Create a mask to check if all values are zero (i.e., no nonzero values in the column)
mask_all_zero = (rotated_matrix == 0).all(axis=2)

# Use advanced indexing to get the first nonzero values
first_nonzero_values = rotated_matrix[np.arange(rotated_matrix.shape[0])[:, None],  # X indices
                              np.arange(rotated_matrix.shape[1]),          # Y indices
                              first_nonzero_idx]                    # Z indices

# Set positions where all Z values are zero to a default (e.g., 0 or NaN)
first_nonzero_values[mask_all_zero] = 0  # You can also use np.nan if needed

# Print the result
print("First nonzero values along Z-axis:")
print(first_nonzero_values)