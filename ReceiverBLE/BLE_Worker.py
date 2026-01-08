import asyncio
import logging
import UltraFastDatabaseManager
from bleak import BleakClient, BleakScanner
from PyQt6.QtCore import pyqtSignal, QThread
from datetime import datetime
from IMUState import IMUState

# --- BLE WORKER THREAD ---
class BLEWorkerThread(QThread):
    """Thread to run the async BLE connection without blocking the GUI"""

    imu_data_signal = pyqtSignal(object)  # roll, pitch, yaw
    ble_status_signal = pyqtSignal(str, str)  # status_message, color_code
    connection_state_signal = pyqtSignal(bool)  # True=connected, False=disconnected

    def __init__(self, device_address, db_table_name, db_history_table_name):
        super().__init__()
        self.device_address = device_address
        self.db_table_name = db_table_name
        self.db_history_table_name = db_history_table_name
        self.db_manager = None  # Will be created in the worker thread's event loop
        self.running = False
        self.should_connect = False
        self.loop = None
        self.client = None
        self.is_connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3  # Max attempts before giving up

        self.imu = IMUState()

    def run(self):
        """Run the asyncio event loop in this thread"""
        self.running = True
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main())
        except Exception as e:
            logging.error(f"BLE Worker error: {e}")
            self.ble_status_signal.emit(f"Error: {str(e)}", "red")
            self.connection_state_signal.emit(False)
        finally:
            if self.loop:
                self.loop.close()

    def request_connect(self):
        """Request connection to BLE device"""
        self.should_connect = True
        self.connection_attempts = 0  # Reset attempts on new connection request

    def request_disconnect(self):
        """Request disconnection from BLE device"""
        self.should_connect = False

    async def check_bluetooth_available(self):
        """Check if Bluetooth adapter is available"""
        try:
            # Try to discover devices briefly to test if BT is working
            devices = await BleakScanner.discover(timeout=2.0)
            return True
        except Exception as e:
            logging.error(f"Bluetooth check failed: {e}")
            return False

    def notification_handler(self, sender, data):
        """Handler for BLE notifications - modified to emit Qt signals"""
        try:
            text = data.decode('utf-8').strip()
            parts = [p.strip() for p in text.split(',')]

            if len(parts) >= 12:
                BLECounter = int(parts[0])
                reportSecond = int(parts[1])
                angleReportLoopCount = int(parts[2])
                transferredLoopCount = int(parts[3])
                status = int(parts[4])

                yaw = float(parts[5].replace("YPR=", ""))
                pitch = float(parts[6])
                roll = float(parts[7])

                self.imu.update(roll, pitch, yaw)

                quat_r = float(parts[8].replace("Q=", ""))
                quat_i = float(parts[9])
                quat_j = float(parts[10])
                quat_k = float(parts[11])

                data_dict = {
                    'ble_counter': BLECounter,
                    'report_second': reportSecond,
                    'angle_report_loop_count': angleReportLoopCount,
                    'transferred_loop_count': transferredLoopCount,
                    'status': status,
                    'yaw': yaw,
                    'yaw_delta': self.imu.yaw_delta,
                    'yaw_calibrated': self.imu.yaw_calibrated,
                    'pitch': pitch,
                    'roll': roll,
                    'quat_r': quat_r,
                    'quat_i': quat_i,
                    'quat_j': quat_j,
                    'quat_k': quat_k,
                }

                timestamp = datetime.now()

                logging.info(
                    f"[{self.db_table_name}] loop={BLECounter}, {reportSecond}, {angleReportLoopCount}, {transferredLoopCount} "
                    f"status={status}, "
                    f"YPR=({yaw:.2f}, {pitch:.2f}, {roll:.2f}), "
                    f"Quat=({quat_r:.3f}, {quat_i:.3f}, {quat_j:.3f}, {quat_k:.3f})"
                )

                # Emit signal to update GUI (thread-safe)
                self.imu_data_signal.emit(self.imu)

                if self.db_manager:
                    # Fast async update of current record - already on the right loop!f
                    asyncio.create_task(
                        self.db_manager.update_current(data_dict, timestamp)
                    )

        except Exception as e:
            logging.error(f"Notification error: {e}")
            self.ble_status_signal.emit(f"Notification error: {str(e)}", "orange")

    def request_yaw_reset(self):

        self.imu.calibrate_yaw()

    async def async_main(self):
        """Async main function to connect to BLE device"""
        self.ble_status_signal.emit("Ready to connect", "orange")
        self.connection_state_signal.emit(False)

        try:
            # Create DB manager on THIS thread's event loop
            DB_CONFIG = UltraFastDatabaseManager.DB_CONFIG
            self.db_manager = UltraFastDatabaseManager.UltraFastDatabaseManager(
                DB_CONFIG, self.db_table_name, self.db_history_table_name
            )

            # Create the pool on this event loop
            await self.db_manager.create_pool()
            logging.info(f"[{self.db_table_name}] DB pool created on worker thread loop")

            while self.running:
                if self.should_connect and not self.is_connected:
                    await self.connect_to_device()
                elif not self.should_connect and self.is_connected:
                    await self.disconnect_from_device()
                await asyncio.sleep(0.1)

        except Exception as e:
            self.ble_status_signal.emit(f"Connection error: {str(e)}", "red")
            self.connection_state_signal.emit(False)
            logging.error(f"BLE connection error: {e}")
        finally:
            if self.is_connected:
                await self.disconnect_from_device()

            # Close the DB pool on the SAME loop it was created on
            if self.db_manager and self.db_manager.pool is not None:
                await self.db_manager.close_pool()
                logging.info(f"[{self.db_table_name}] DB pool closed")

    async def connect_to_device(self):
        """Connect to the BLE device"""
        try:
            self.connection_attempts += 1

            # Check if we've exceeded max attempts
            if self.connection_attempts > self.max_connection_attempts:
                self.ble_status_signal.emit(
                    "Connection failed: Max attempts reached. Check Bluetooth adapter.",
                    "red"
                )
                self.should_connect = False  # Stop trying to connect
                self.connection_state_signal.emit(False)
                self.is_connected = False
                return

            self.ble_status_signal.emit(
                f"Connecting... (Attempt {self.connection_attempts}/{self.max_connection_attempts})",
                "orange"
            )

            # First check if Bluetooth is available
            if self.connection_attempts == 1:  # Only check on first attempt
                bt_available = await self.check_bluetooth_available()
                if not bt_available:
                    self.ble_status_signal.emit(
                        "Bluetooth adapter not found. Please plug it in and try again.",
                        "red"
                    )
                    self.should_connect = False  # Stop trying
                    self.connection_state_signal.emit(False)
                    self.is_connected = False
                    return

            self.client = BleakClient(self.device_address)
            await self.client.connect()

            if not self.client.is_connected:
                self.ble_status_signal.emit("Failed to connect", "red")
                self.connection_state_signal.emit(False)
                self.is_connected = False
                return

            # Success! Reset attempts counter
            self.connection_attempts = 0
            self.is_connected = True
            self.ble_status_signal.emit("Connected", "green")
            self.connection_state_signal.emit(True)

            await self.client.start_notify(
                UltraFastDatabaseManager.CHARACTERISTIC_UUID_1,
                self.notification_handler
            )
        except Exception as e:
            error_msg = str(e)

            # Check for specific Bluetooth adapter errors
            if "Failed to start scanner" in error_msg or "Bluetooth" in error_msg:
                self.ble_status_signal.emit(
                    f"Bluetooth adapter error. Please check if adapter is plugged in.",
                    "red"
                )
                self.should_connect = False  # Stop trying on adapter errors
            else:
                self.ble_status_signal.emit(f"Connection failed: {error_msg}", "red")

            self.connection_state_signal.emit(False)
            self.is_connected = False
            logging.error(f"Failed to connect (attempt {self.connection_attempts}): {e}")

    async def disconnect_from_device(self):
        """Disconnect from the BLE device"""
        try:
            if self.client and self.client.is_connected:
                self.ble_status_signal.emit("Disconnecting...", "orange")
                await self.client.stop_notify(
                    UltraFastDatabaseManager.CHARACTERISTIC_UUID_1
                )
                await self.client.disconnect()

            self.is_connected = False
            self.client = None
            self.connection_attempts = 0  # Reset attempts on disconnect
            self.ble_status_signal.emit("Disconnected", "red")
            self.connection_state_signal.emit(False)
        except Exception as e:
            self.ble_status_signal.emit(f"Disconnect error: {str(e)}", "orange")
            logging.error(f"Failed to disconnect: {e}")
            self.is_connected = False
            self.connection_state_signal.emit(False)

    def stop(self):
        """Stop the BLE worker thread"""
        self.running = False
        self.should_connect = False