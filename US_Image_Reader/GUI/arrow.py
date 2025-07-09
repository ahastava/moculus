from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
import numpy as np

# Sample data (replace with actual data)
accel_rel = np.array([[0.1, 0.0, 9.7]])  # Accelerometer data (ax, ay, az)
gyro_rel = np.array([[0.1, 0.0, 0.0]])  # Gyroscope data (gx, gy, gz)

# Initialize the Madgwick filter with a sample period
madgwick = Madgwick(sampleperiod=1/100.0)  # Assuming 100Hz data rate

# Process the data using update_imu() to update the internal filter state
quaternions = []
for acc, gyr in zip(accel_rel, gyro_rel):
    madgwick.updateIMU(acc=acc, gyr=gyr)  # Update filter with accelerometer and gyroscope data
    quaternions.append(madgwick.Q.copy())  # Store quaternion for each iteration

# Convert the quaternions to Euler angles (roll, pitch, yaw)
quaternions = np.array(quaternions)
eulers_rad = q2euler(quaternions)  # Convert quaternion to Euler angles (radians)
eulers_deg = np.degrees(eulers_rad)  # Convert radians to degrees

# Print the roll, pitch, and yaw for each step
for i, (roll, pitch, yaw) in enumerate(eulers_deg):
    print(f"Step {i}: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")