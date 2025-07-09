from ahrs.filters import Madgwick
import numpy as np

accel_rel = np.array(
    [[-0.03, -0.79, 9.48]],  # IMU1 (fixed on car)
)

gyro_rel = np.array(
    [[0.06, 0.19, 0.00]],  # IMU1 (fixed on car)
)

mag_rel = np.array(
    [[13.38, 8.88, -48.38]],  # IMU1 (fixed on car)
)

# Orientation (Euler): Yaw=288.94 Pitch=0.19 Roll=4.75 degrees
# Linear Accel (m/s^2): X=-0.03 Y=-0.79 Z=9.48
# Gyroscope (deg/s): X=-0.06 Y=0.19 Z=0.00
# Magnetometer (µT): X=13.38 Y=8.88 Z=-48.38

print( accel_rel)
print( gyro_rel)

#print( zip(accel_rel, gyro_rel))
madgwick = Madgwick(gyr=gyro_rel, acc=accel_rel, mag=mag_rel)

print(madgwick)

from ahrs.common.orientation import q2euler

eulers_rad = q2euler(madgwick.Q[0])  # Convert quaternion to Euler angles (radians)
eulers_deg = np.degrees(eulers_rad)  # Convert radians to degrees

(roll, pitch, yaw) = eulers_deg
print(f"Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")

# # Print the roll, pitch, and yaw for each step
# for i, (roll, pitch, yaw) in enumerate(eulers_deg):
#     print(f"Step {i}: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")

# import ahfs
#
# QQ = ahrs.QuaternionArray(madgwick.Q)
# euler_angles = QQ.to_angles()