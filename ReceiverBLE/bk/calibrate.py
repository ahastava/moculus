from bk import mysql_interface
import copy
from dataclasses import dataclass


@dataclass
class IMU:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    quat_r: float = 0.0
    quat_i: float = 0.0
    quat_j: float = 0.0
    quat_k: float = 0.0


class Calibrate:
    def __init__(self):
        super().__init__()
        # Initial calibration values (t=0)
        self.probe_imu_0 = IMU()
        self.car_imu_0 = IMU()

        # Previous frame values
        self.car_imu_prev = IMU()
        self.probe_imu_prev = IMU()

        # Current frame values
        self.car_imu_current = IMU()
        self.probe_imu_current = IMU()

        # Calibrated results
        self.probe_imu_prev_calibrated = IMU()
        self.probe_imu_calibrated_1 = IMU()  # Method 1: Cumulative
        self.probe_imu_calibrated_2 = IMU()  # Method 2: Incremental

    def set_first_data(self):
        """Load initial calibration data from database"""
        # Get probe IMU initial data
        data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.probe_imu_history ORDER BY ble_counter ASC LIMIT 1",
            []
        )
        print("Probe t=0:", data)
        if len(data) > 0:
            self.probe_imu_0 = IMU(
                data[0]['yaw'], data[0]['pitch'], data[0]['roll'],
                data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
            )

        # Get car IMU initial data
        data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.car_imu_history ORDER BY ble_counter ASC LIMIT 1",
            []
        )
        print("Car t=0:", data)
        if len(data) > 0:
            self.car_imu_0 = IMU(
                data[0]['yaw'], data[0]['pitch'], data[0]['roll'],
                data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
            )

        # Initialize previous and current with t=0 values
        self.car_imu_prev = copy.deepcopy(self.car_imu_0)
        self.probe_imu_prev = copy.deepcopy(self.probe_imu_0)
        self.car_imu_current = copy.deepcopy(self.car_imu_0)
        self.probe_imu_current = copy.deepcopy(self.probe_imu_0)

        # Initialize calibrated values
        self.probe_imu_prev_calibrated = copy.deepcopy(self.probe_imu_0)
        self.probe_imu_calibrated_1 = copy.deepcopy(self.probe_imu_0)
        self.probe_imu_calibrated_2 = copy.deepcopy(self.probe_imu_0)


    def reset_history_data(self):
        """Load initial calibration data from database"""
        # Get probe IMU initial

        d = mysql_interface.DBWorker()

        d.db_raw_select("SET SQL_SAFE_UPDATES = 0;")
        d.db_raw_select("DELETE FROM probe_imu_history;")
        d.db_raw_select("DELETE FROM car_imu_history;")
        d.db_raw_select("SET SQL_SAFE_UPDATES = 1;")


    def wrap180(self, angle):
        """Wrap angle to (-180, 180]"""
        return (angle + 180) % 360 - 180

    def calibrate_method1_cumulative(self):
        """
        Method 1: Cumulative/Absolute correction
        Formula: angle_corrected = (angle_hand - angle_car) - (angle_hand_t0 - angle_car_t0)

        This removes the initial offset and car's rotation from hand's orientation
        """
        # Yaw correction
        yaw_relative = self.wrap180(
            (self.probe_imu_current.yaw - self.car_imu_current.yaw) -
            (self.probe_imu_0.yaw - self.car_imu_0.yaw)
        )

        # Pitch correction
        pitch_relative = self.wrap180(
            (self.probe_imu_current.pitch - self.car_imu_current.pitch) -
            (self.probe_imu_0.pitch - self.car_imu_0.pitch)
        )

        # Roll correction
        roll_relative = self.wrap180(
            (self.probe_imu_current.roll - self.car_imu_current.roll) -
            (self.probe_imu_0.roll - self.car_imu_0.roll)
        )

        # Store calibrated values
        self.probe_imu_calibrated_1.yaw = self.wrap180(yaw_relative + self.probe_imu_0.yaw)
        self.probe_imu_calibrated_1.pitch = self.wrap180(pitch_relative + self.probe_imu_0.pitch)
        self.probe_imu_calibrated_1.roll = self.wrap180(roll_relative + self.probe_imu_0.roll)

        return self.probe_imu_calibrated_1

    def calibrate_method2_incremental(self):
        """
        Method 2: Incremental/Delta correction
        Formula: angle_corrected = angle_corrected_prev + (delta_hand - delta_car)

        This accumulates the difference in deltas over time
        """
        # Calculate deltas from previous frame
        delta_car_yaw = self.wrap180(self.car_imu_current.yaw - self.car_imu_prev.yaw)
        delta_car_pitch = self.wrap180(self.car_imu_current.pitch - self.car_imu_prev.pitch)
        delta_car_roll = self.wrap180(self.car_imu_current.roll - self.car_imu_prev.roll)

        delta_hand_yaw = self.wrap180(self.probe_imu_current.yaw - self.probe_imu_prev.yaw)
        delta_hand_pitch = self.wrap180(self.probe_imu_current.pitch - self.probe_imu_prev.pitch)
        delta_hand_roll = self.wrap180(self.probe_imu_current.roll - self.probe_imu_prev.roll)

        # Accumulate compensated deltas
        self.probe_imu_calibrated_2.yaw = self.wrap180(
            self.probe_imu_prev_calibrated.yaw + (delta_hand_yaw - delta_car_yaw)
        )
        self.probe_imu_calibrated_2.pitch = self.wrap180(
            self.probe_imu_prev_calibrated.pitch + (delta_hand_pitch - delta_car_pitch)
        )
        self.probe_imu_calibrated_2.roll = self.wrap180(
            self.probe_imu_prev_calibrated.roll + (delta_hand_roll - delta_car_roll)
        )

        # Update previous calibrated for next iteration
        self.probe_imu_prev_calibrated = copy.deepcopy(self.probe_imu_calibrated_2)

        return self.probe_imu_calibrated_2

    def update_current_data(self):
        """Update current frame data from new readings"""
        # Store previous values
        self.car_imu_prev = copy.deepcopy(self.car_imu_current)
        self.probe_imu_prev = copy.deepcopy(self.probe_imu_current)

        data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.probe_imu ORDER BY ble_counter",
            []
        )
        print("Probe t=0:", data)
        self.probe_imu_current = IMU(
            data[0]['yaw'], data[0]['pitch'], data[0]['roll'],
            data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
        )

        # Get car IMU initial data
        data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.car_imu",
            []
        )
        print("Car t=0:", data)
        self.car_imu_current = IMU(
            data[0]['yaw'], data[0]['pitch'], data[0]['roll'],
            data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
        )


    def update_test_data(self):
        """Update current frame data from new readings"""
        # Store previous values
        self.car_imu_prev = copy.deepcopy(self.car_imu_current)
        self.probe_imu_prev = copy.deepcopy(self.probe_imu_current)

        # roll, yaw, pitch
        self.car_imu_0.yaw = 0
        self.probe_imu_0.yaw = 5

        self.car_imu_prev.yaw = 0
        self.probe_imu_prev.yaw = 5

        self.probe_imu_prev_calibrated.yaw = 5

        self.car_imu_current.yaw = 10
        self.probe_imu_current.yaw = -25

        # roll, yaw, pitch
        self.car_imu_0.pitch = 0
        self.probe_imu_0.pitch = 5

        self.car_imu_prev.pitch = 0
        self.probe_imu_prev.pitch = 5

        self.probe_imu_prev_calibrated.pitch = 5

        self.car_imu_current.pitch = 10
        self.probe_imu_current.pitch = -25

        # roll, yaw, pitch
        self.car_imu_0.roll = 0
        self.probe_imu_0.roll = 5

        self.car_imu_prev.roll = 0
        self.probe_imu_prev.roll = 5

        self.probe_imu_prev_calibrated.roll = 5

        self.car_imu_current.roll = 10
        self.probe_imu_current.roll = -25

        print("\nThis test data should give: Yaw: 15.00°, Pitch: 15.00°, Roll: 15.00° ")

    def run_calibration(self):
        """Run both calibration methods and return results"""
        method1_result = self.calibrate_method1_cumulative()
        method2_result = self.calibrate_method2_incremental()

        print(f"\nMethod 1 (Cumulative):")
        print(f"  Yaw: {method1_result.yaw:.2f}°, Pitch: {method1_result.pitch:.2f}°, Roll: {method1_result.roll:.2f}°")

        print(f"\nMethod 2 (Incremental):")
        print(f"  Yaw: {method2_result.yaw:.2f}°, Pitch: {method2_result.pitch:.2f}°, Roll: {method2_result.roll:.2f}°")

        return method1_result, method2_result

    def process_all_data(self):
        """Process all historical data from database"""
        # Get all probe data
        probe_data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.probe_imu_history ORDER BY ble_counter ASC",
            []
        )

        # Get all car data
        car_data = mysql_interface.db_select(
            None,
            "SELECT * FROM ble_receiver.car_imu_history ORDER BY ble_counter ASC",
            []
        )

        results_method1 = []
        results_method2 = []

        # Process each frame
        for i in range(len(probe_data)):
            self.update_current_data(probe_data[i], car_data[i])
            m1, m2 = self.run_calibration()
            results_method1.append(copy.deepcopy(m1))
            results_method2.append(copy.deepcopy(m2))

        return results_method1, results_method2


# Usage example
if __name__ == "__main__":
    calibrator = Calibrate()
    calibrator.set_first_data()
    #calibrator.reset_history_data()
    # Example: Process a single new reading
    calibrator.update_current_data()
    calibrator.update_test_data()
    calibrator.run_calibration()

    # Or process all historical data
    # results_m1, results_m2 = calibrator.process_all_data()