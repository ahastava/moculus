import numpy as np

# Example inputs
yaw_car  = np.array([0, 10, 20])   # degrees
#yaw_hand = np.array([5, 15, 25])   # degrees
yaw_hand = np.array([5, 25, 35])   # degrees

def wrap180(angle):
    """Wrap angle to (-180, 180]"""
    return (angle + 180) % 360 - 180

# --- Method A: Cumulative correction (remove initial offset)
yaw_corrected_cumulative = wrap180((yaw_hand - yaw_car) - wrap180(yaw_hand[0] - yaw_car[0]))

# add back the initial hand orientation
yaw_corrected_cumulative_final = wrap180(yaw_corrected_cumulative + yaw_hand[0])

# --- Method B: Incremental correction (delta accumulation)
yaw_corrected_incremental = np.zeros_like(yaw_car, dtype=float)

for t in range(1, len(yaw_car)):
    delta_car  = wrap180(yaw_car[t] - yaw_car[t-1])
    delta_hand = wrap180(yaw_hand[t] - yaw_hand[t-1])
    yaw_corrected_incremental[t] = wrap180(yaw_corrected_incremental[t - 1] + (delta_hand - delta_car))

# add back the initial hand orientation
yaw_corrected_incremental_final = wrap180(yaw_corrected_incremental + yaw_hand[0])

# Print results
print("yaw_car: ", yaw_car)
print("yaw_hand:", yaw_hand)
print("Cumulative correction:", yaw_corrected_cumulative, yaw_corrected_cumulative_final)
print("Incremental correction:", yaw_corrected_incremental, yaw_corrected_incremental_final)