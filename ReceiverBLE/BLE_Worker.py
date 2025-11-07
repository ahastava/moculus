import asyncio
import logging
from bk import dual_imu_handler
from bleak import BleakClient
from PyQt6.QtCore import pyqtSignal, QThread
from datetime import datetime


# --- BLE WORKER THREAD ---
class BLEWorkerThread(QThread):
    """Thread to run the async BLE connection without blocking the GUI"""

    imu_data_signal = pyqtSignal(float, float, float)  # roll, pitch, yaw
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

    def request_disconnect(self):
        """Request disconnection from BLE device"""
        self.should_connect = False

    def notification_handler(self, sender, data):
        """Handler for BLE notifications - modified to emit Qt signals"""
        try:
            text = data.decode('utf-8').strip()
            parts = [p.strip() for p in text.split(',')]
            print("1111111111")
            if len(parts) >= 12:
                BLECounter = int(parts[0])
                reportSecond = int(parts[1])
                angleReportLoopCount = int(parts[2])
                transferredLoopCount = int(parts[3])
                status = int(parts[4])

                yaw = float(parts[5].replace("YPR=", ""))
                pitch = float(parts[6])
                roll = float(parts[7])

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
                self.imu_data_signal.emit(float(roll), float(pitch), float(yaw))

                if self.db_manager:
                    # Fast async update of current record - already on the right loop!
                    asyncio.create_task(
                        self.db_manager.update_current(data_dict, timestamp)
                    )

        except Exception as e:
            logging.error(f"Notification error: {e}")
            self.ble_status_signal.emit(f"Notification error: {str(e)}", "orange")

    async def async_main(self):
        """Async main function to connect to BLE device"""
        self.ble_status_signal.emit("Ready to connect", "orange")
        self.connection_state_signal.emit(False)

        try:
            # Create DB manager on THIS thread's event loop
            DB_CONFIG = dual_imu_handler.DB_CONFIG
            self.db_manager = dual_imu_handler.UltraFastDatabaseManager(
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
            self.ble_status_signal.emit("Connecting...", "orange")
            self.client = BleakClient(self.device_address)
            await self.client.connect()

            if not self.client.is_connected:
                self.ble_status_signal.emit("Failed to connect", "red")
                self.connection_state_signal.emit(False)
                self.is_connected = False
                return

            self.is_connected = True
            self.ble_status_signal.emit("Connected", "green")
            self.connection_state_signal.emit(True)

            await self.client.start_notify(
                dual_imu_handler.CHARACTERISTIC_UUID_1,
                self.notification_handler
            )
        except Exception as e:
            self.ble_status_signal.emit(f"Connection failed: {str(e)}", "red")
            self.connection_state_signal.emit(False)
            self.is_connected = False
            logging.error(f"Failed to connect: {e}")

    async def disconnect_from_device(self):
        """Disconnect from the BLE device"""
        try:
            if self.client and self.client.is_connected:
                self.ble_status_signal.emit("Disconnecting...", "orange")
                await self.client.stop_notify(
                    dual_imu_handler.CHARACTERISTIC_UUID_1
                )
                await self.client.disconnect()

            self.is_connected = False
            self.client = None
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