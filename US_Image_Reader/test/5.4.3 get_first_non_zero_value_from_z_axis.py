import numpy as np

# Example 3D volume: 5x5x5 for demonstration
matrix = np.array([
    [[ 0,  0,  0,  4,  5],   # First nonzero in Z: 4
     [ 0,  0,  0,  0,  0],   # All zero (should return 0 or NaN)
     [ 0,  0, 10, 20, 30],   # First nonzero in Z: 10
     [ 0,  1,  2,  3,  4],   # First nonzero in Z: 1
     [ 0,  0,  0,  7,  8]],  # First nonzero in Z: 7

    [[ 0,  0,  0,  0,  0],
     [ 0,  0,  9,  0,  0],
     [ 0,  0,  0,  0,  0],
     [ 0,  3,  0,  0,  0],
     [ 0,  0,  6,  0,  0]]
])

# Get first nonzero index along Z-axis
first_nonzero_idx = np.argmax(matrix != 0, axis=2)  # Indices where first nonzero appears

# Create a mask to check if all values are zero (i.e., no nonzero values in the column)
mask_all_zero = (matrix == 0).all(axis=2)

# Use advanced indexing to get the first nonzero values
first_nonzero_values = matrix[np.arange(matrix.shape[0])[:, None],  # X indices
                              np.arange(matrix.shape[1]),          # Y indices
                              first_nonzero_idx]                    # Z indices

# Set positions where all Z values are zero to a default (e.g., 0 or NaN)
first_nonzero_values[mask_all_zero] = 0  # You can also use np.nan if needed

# Print the result
print("First nonzero values along Z-axis:")
print(first_nonzero_values)