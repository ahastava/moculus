import pymysql  # Use pymysql instead of mysql.connector
import config
import copy
from PyQt6.QtCore import QThread, pyqtSignal
import time
import traceback
import logging
from IMUState import IMUState
from datetime import datetime

class CalibrateWorker(QThread):
    """Worker thread for calibration operations with database access"""

    # Signals
    status_signal = pyqtSignal(str, str)  # message, color
    calibration_result_signal = pyqtSignal(object, object)  # method1_result, method2_result
    reset_complete_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.DB_HOST = config.DB_CONFIG['host']
        self.DB_PASS = config.DB_CONFIG['password']
        self.DB_NAME = config.DB_CONFIG['db']
        self.DB_USER = config.DB_CONFIG['user']

        logging.info(f"DB Config loaded: host={self.DB_HOST}, user={self.DB_USER}, db={self.DB_NAME}")

        # Initial calibration values (t=0)
        self.probe_imu_0 = IMUState()
        self.car_imu_0 = IMUState()

        # Previous frame values
        self.car_imu_prev = IMUState()
        self.probe_imu_prev = IMUState()

        # Current frame values
        self.car_imu_current = IMUState()
        self.probe_imu_current = IMUState()

        # Calibrated results
        self.probe_imu_prev_calibrated = IMUState()
        self.probe_imu_calibrated_1 = IMUState()  # Method 1: Cumulative
        self.probe_imu_calibrated_2 = IMUState()  # Method 2: Incremental

        # Operation queue
        self.pending_operation = None
        self.running = False

        self.continuous_mode = False

    def run(self):
        """Thread execution - keeps running and processes operations"""
        self.running = True

        while self.running:
            if self.pending_operation == "reset":
                self.pending_operation = None
                self._do_reset_history()
            elif self.pending_operation == "calibrate":
                self.pending_operation = None
                self._do_calibration()
            else:
                # Sleep a bit to avoid busy waiting
                time.sleep(0.1)

    def request_reset(self):
        """Request to reset history tables"""
        logging.info("Reset requested")
        self.pending_operation = "reset"

    def request_calibration(self):
        """Request to run calibration"""
        logging.info("Calibration requested")
        self.pending_operation = "calibrate"

    def stop(self):
        """Stop the worker thread"""
        self.running = False

    def _get_db_connection(self):
        """Create a new database connection using PyMySQL (thread-safe)"""
        try:
            logging.info(f"Attempting DB connection to {self.DB_HOST} as {self.DB_USER}...")
            con = pymysql.connect(
                host=self.DB_HOST,
                user=self.DB_USER,
                password=self.DB_PASS,
                database=self.DB_NAME,
                connect_timeout=10,
                cursorclass=pymysql.cursors.DictCursor  # Return dict results
            )
            logging.info("DB connection successful!")
            return con
        except pymysql.Error as err:
            error_msg = f"MySQL Error: {err}"
            logging.info(error_msg)
            self.error_signal.emit(error_msg)
            return None
        except Exception as e:
            error_msg = f"DB connection failed: {e}"
            logging.info(error_msg)
            logging.info(traceback.format_exc())
            self.error_signal.emit(error_msg)
            return None

    def _db_execute(self, sql, data=None):
        """Execute a database query"""
        con = None
        cursor = None
        try:
            logging.info(f"Executing SQL: {sql[:100]}...")  # Print first 100 chars
            con = self._get_db_connection()
            if not con:
                print("Failed to get DB connection")
                return None

            logging.info("Got DB connection, creating cursor...")
            cursor = con.cursor()

            logging.info("Executing query...")
            if data:
                cursor.execute(sql, data)
            else:
                cursor.execute(sql)

            # If it's a SELECT, fetch results
            if sql.strip().lower().startswith("select"):
                result = cursor.fetchall()
                logging.info(f"Query returned {len(result)} rows")
            else:
                # For DELETE/UPDATE/INSERT or SET commands, commit changes
                con.commit()
                result = f"OK ({cursor.rowcount} rows affected)"
                logging.info(f"Query result: {result}")

            return result

        except pymysql.Error as err:
            error_msg = f"MySQL query error: {err}"
            logging.info(error_msg)
            self.error_signal.emit(error_msg)
            return None
        except Exception as e:
            error_msg = f"DB query failed: {e}"
            logging.info(error_msg)
            logging.info(traceback.format_exc())
            self.error_signal.emit(error_msg)
            return None
        finally:
            if cursor:
                cursor.close()
                logging.info("Cursor closed")
            if con:
                con.close()
                logging.info("Connection closed")

    def _do_reset_history(self):
        """Reset history tables (runs in thread)"""
        try:
            logging.info("Starting reset operation...")
            self.status_signal.emit("Resetting history tables...", "orange")

            self._db_execute("SET SQL_SAFE_UPDATES = 0;")
            result1 = self._db_execute("TRUNCATE TABLE probe_imu_history;")
            logging.info(f"Probe delete result: {result1}")

            result1 = self._db_execute("""
                            INSERT INTO probe_imu_history
                            SELECT * FROM probe_imu
                            LIMIT 1;""")
            logging.info(f"Probe delete result: {result1}")



            result2 = self._db_execute("TRUNCATE TABLE car_imu_history;")
            logging.info(f"Car delete result: {result2}")

            result2 = self._db_execute("""
                            INSERT INTO car_imu_history
                            SELECT * FROM car_imu
                            LIMIT 1;""")
            logging.info(f"Car delete result: {result2}")


            self._db_execute("SET SQL_SAFE_UPDATES = 1;")

            self.status_signal.emit("History tables reset", "green")
            self.reset_complete_signal.emit()
            logging.info("Reset complete")

            self._set_first_data()

        except Exception as e:
            error_msg = f"Reset failed: {e}"
            self.error_signal.emit(error_msg)
            self.status_signal.emit(error_msg, "red")
            logging.info(error_msg)

    def _set_first_data(self):
        """Load initial calibration data from database"""
        # Get probe IMU initial data
        data = self._db_execute(
            "SELECT * FROM ble_receiver.probe_imu_history ORDER BY ble_counter ASC LIMIT 1"
        )

        if data and len(data) > 0:
            self.probe_imu_0 = IMUState()

            self.probe_imu_0.update(
                data[0]['roll'],
                data[0]['pitch'],
                data[0]['yaw_calibrated'],
                data[0]['quat_r'],
                data[0]['quat_i'],
                data[0]['quat_j'],
                data[0]['quat_k']
            )

            logging.info(f"Probe t=0: {data}")

        # Get car IMU initial data
        data = self._db_execute(
            "SELECT * FROM ble_receiver.car_imu_history ORDER BY ble_counter ASC LIMIT 1"
        )

        if data and len(data) > 0:
            # self.car_imu_0 = IMUState(
            #     data[0]['yaw_calibrated'], data[0]['pitch'], data[0]['roll'],
            #     data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
            # )
            self.car_imu_0 = IMUState()

            self.car_imu_0.update(
                data[0]['roll'],
                data[0]['pitch'],
                data[0]['yaw_calibrated'],
                data[0]['quat_r'],
                data[0]['quat_i'],
                data[0]['quat_j'],
                data[0]['quat_k']
            )
            logging.info(f"Car t=0: {data}")

        # Initialize previous and current with t=0 values
        self.car_imu_prev = copy.deepcopy(self.car_imu_0)
        self.probe_imu_prev = copy.deepcopy(self.probe_imu_0)
        self.car_imu_current = copy.deepcopy(self.car_imu_0)
        self.probe_imu_current = copy.deepcopy(self.probe_imu_0)

        # Initialize calibrated values
        self.probe_imu_prev_calibrated = copy.deepcopy(self.probe_imu_0)
        self.probe_imu_calibrated_1 = copy.deepcopy(self.probe_imu_0)
        self.probe_imu_calibrated_2 = copy.deepcopy(self.probe_imu_0)

    def _read_current_data(self):
        """Update current frame data from new readings"""
        # Store previous values
        self.car_imu_prev = copy.deepcopy(self.car_imu_current)
        self.probe_imu_prev = copy.deepcopy(self.probe_imu_current)

        # Get probe IMU current data
        data = self._db_execute(
            "SELECT * FROM ble_receiver.probe_imu ORDER BY ble_counter"
        )

        if data and len(data) > 0:
            # self.probe_imu_current = IMU(
            #     data[0]['yaw_calibrated'], data[0]['pitch'], data[0]['roll'],
            #     data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
            # )
            self.probe_imu_current = IMUState()

            self.probe_imu_current.update(
                data[0]['roll'],
                data[0]['pitch'],
                data[0]['yaw_calibrated'],
                data[0]['quat_r'],
                data[0]['quat_i'],
                data[0]['quat_j'],
                data[0]['quat_k']
            )
            logging.info(f"Probe current: {data}")

        # Get car IMU current data
        data = self._db_execute(
            "SELECT * FROM ble_receiver.car_imu"
        )

        if data and len(data) > 0:
            # self.car_imu_current = IMU(
            #     data[0]['yaw_calibrated'], data[0]['pitch'], data[0]['roll'],
            #     data[0]['quat_r'], data[0]['quat_i'], data[0]['quat_j'], data[0]['quat_k']
            # )
            self.car_imu_current = IMUState()

            self.car_imu_current.update(
                data[0]['roll'],
                data[0]['pitch'],
                data[0]['yaw_calibrated'],
                data[0]['quat_r'],
                data[0]['quat_i'],
                data[0]['quat_j'],
                data[0]['quat_k']
            )
            logging.info(f"Car current: {data}")

    def _write_calibrated_result(self, imu: IMUState, imu2: IMUState):
        #Write calibrated IMU (per method) into calibrated_imu table.

        try:
            sql = f"""Update calibrated_imu set
                    roll = %s,
                    pitch = %s,
                    yaw = %s,
                    roll_2 = %s,
                    pitch_2 = %s,
                    yaw_2 = %s,
                    updated_at = %s
                    WHERE idx = 1
                    """
            self._db_execute(sql, (imu.roll, imu.pitch, imu.yaw, imu2.roll, imu2.pitch, imu2.yaw, datetime.now() ))
        except Exception as e:
            logging.error(f"Failed to write calibrated result to DB: {e}")

    def _wrap180(self, angle):
        """Wrap angle to (-180, 180]"""
        return (angle + 180) % 360 - 180

    def _calibrate_method1_cumulative(self):
        """Method 1: Cumulative/Absolute correction"""
        yaw_relative = self._wrap180(
            (self.probe_imu_current.yaw - self.car_imu_current.yaw) -
            (self.probe_imu_0.yaw - self.car_imu_0.yaw)
        )

        pitch_relative = self._wrap180(
            (self.probe_imu_current.pitch - self.car_imu_current.pitch) -
            (self.probe_imu_0.pitch - self.car_imu_0.pitch)
        )

        roll_relative = self._wrap180(
            (self.probe_imu_current.roll - self.car_imu_current.roll) -
            (self.probe_imu_0.roll - self.car_imu_0.roll)
        )

        self.probe_imu_calibrated_1.yaw = self._wrap180(yaw_relative + self.probe_imu_0.yaw)
        self.probe_imu_calibrated_1.pitch = self._wrap180(pitch_relative + self.probe_imu_0.pitch)
        self.probe_imu_calibrated_1.roll = self._wrap180(roll_relative + self.probe_imu_0.roll)

        return self.probe_imu_calibrated_1

    def _calibrate_method2_incremental(self):
        """Method 2: Incremental/Delta correction"""
        delta_car_yaw = self._wrap180(self.car_imu_current.yaw - self.car_imu_prev.yaw)
        delta_car_pitch = self._wrap180(self.car_imu_current.pitch - self.car_imu_prev.pitch)
        delta_car_roll = self._wrap180(self.car_imu_current.roll - self.car_imu_prev.roll)

        delta_hand_yaw = self._wrap180(self.probe_imu_current.yaw - self.probe_imu_prev.yaw)
        delta_hand_pitch = self._wrap180(self.probe_imu_current.pitch - self.probe_imu_prev.pitch)
        delta_hand_roll = self._wrap180(self.probe_imu_current.roll - self.probe_imu_prev.roll)

        self.probe_imu_calibrated_2.yaw = self._wrap180(
            self.probe_imu_prev_calibrated.yaw + (delta_hand_yaw - delta_car_yaw)
        )
        self.probe_imu_calibrated_2.pitch = self._wrap180(
            self.probe_imu_prev_calibrated.pitch + (delta_hand_pitch - delta_car_pitch)
        )
        self.probe_imu_calibrated_2.roll = self._wrap180(
            self.probe_imu_prev_calibrated.roll + (delta_hand_roll - delta_car_roll)
        )

        self.probe_imu_prev_calibrated = copy.deepcopy(self.probe_imu_calibrated_2)

        return self.probe_imu_calibrated_2

    def _do_calibration(self):
        """Run calibration (runs in thread)"""
        try:
            print("Starting calibration...")
            self.status_signal.emit("Running calibration...", "orange")

            # Load initial data
            self._set_first_data()

            # Update current data
            self._read_current_data()

            while self.continuous_mode:
                self._read_current_data()

                method1_result = self._calibrate_method1_cumulative()
                method2_result = self._calibrate_method2_incremental()

                self.calibration_result_signal.emit(
                    copy.deepcopy(method1_result),
                    copy.deepcopy(method2_result)
                )

                self._write_calibrated_result(method1_result, method2_result)

                time.sleep(0.1)  # 10 Hz update

            self.status_signal.emit("Calibration stopped", "yellow")
            logging.info("Calibration loop ended.")

        except Exception as e:
            error_msg = f"Calibration failed: {e}"
            self.error_signal.emit(error_msg)
            self.status_signal.emit(error_msg, "red")
            logging.info(error_msg)
            import traceback
            traceback.print_exc()

    def toggle_continuous(self):
        """Toggle continuous calibration mode"""
        self.continuous_mode = not self.continuous_mode
        if self.continuous_mode:
            logging.info("Continuous calibration started.")
        else:
            logging.info("Continuous calibration stopped.")