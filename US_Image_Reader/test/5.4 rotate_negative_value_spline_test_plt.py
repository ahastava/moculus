import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define x values
x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 10, 30, 20, 0, 0])  # Positive values

# Create a cubic spline interpolator
cs = CubicSpline(x, y)

# Generate fine x values for smooth curve
x_fine = np.linspace(0, 6, 100)
y_fine = cs(x_fine)  # Interpolated values

# Plot
plt.plot(x, y, 'ro', label="Original Points")
plt.plot(x_fine, y_fine, 'b-', label="Cubic Spline")
plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
plt.legend()
plt.title("Cubic Spline Interpolation")
plt.show()