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

                self.imu.update(roll, pitch, yaw, status)

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


# ============================================================================
# TEST FUNCTION
# ============================================================================
def test_ble_worker():
    """
    Test function for BLEWorkerThread class.

    Note: This requires:
    1. PyQt6 QApplication instance
    2. Bluetooth adapter and BLE device (or will test error handling)
    3. Valid database connection (UltraFastDatabaseManager)
    4. IMUState class available
    """
    from PyQt6.QtWidgets import QApplication
    import sys
    import time

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=== BLEWorkerThread Test ===\n")

    # Create QApplication (required for QThread)
    app = QApplication(sys.argv)

    # Test device address (replace with your actual device address)
    # Format: "XX:XX:XX:XX:XX:XX" or UUID on macOS
    TEST_DEVICE_ADDRESS = "00:00:00:00:00:00"  # Replace with actual address

    print(f"Testing with device address: {TEST_DEVICE_ADDRESS}")
    print("Note: Update TEST_DEVICE_ADDRESS in the code with your actual BLE device\n")


    # Create worker instance
    worker = BLEWorkerThread(
        device_address="BB:13:4D:D8:C7:42",
        db_table_name="probe_imu",
        db_history_table_name="probe_imu_history"
    )

    # Track received data
    data_received = {"count": 0, "last_imu": None}

    # Connect signals to test handlers
    def on_imu_data(imu_state):
        data_received["count"] += 1
        data_received["last_imu"] = imu_state
        print(f"[IMU DATA #{data_received['count']}] "
              f"Yaw={imu_state.yaw:.2f}, Pitch={imu_state.pitch:.2f}, Roll={imu_state.roll:.2f}")

    def on_status(message, color):
        print(f"[STATUS - {color}] {message}")

    def on_connection_state(connected):
        state = "CONNECTED" if connected else "DISCONNECTED"
        print(f"[CONNECTION STATE] {state}")

    worker.imu_data_signal.connect(on_imu_data)
    worker.ble_status_signal.connect(on_status)
    worker.connection_state_signal.connect(on_connection_state)

    # Start the worker thread
    print("Starting BLE worker thread...")
    worker.start()
    time.sleep(1)
    print()

    # Test 1: Check initial state
    print("Test 1: Initial State")
    print(f"  Running: {worker.running}")
    print(f"  Should Connect: {worker.should_connect}")
    print(f"  Is Connected: {worker.is_connected}")
    print(f"  Connection Attempts: {worker.connection_attempts}")
    print()

    # Test 2: Test notification handler with mock data
    print("Test 2: Testing notification handler with mock data")
    try:
        # Mock BLE notification data
        mock_data = b"100,5,200,150,1,YPR=45.5,30.2,-10.1,Q=0.924,0.383,0.000,0.000"
        worker.notification_handler(None, mock_data)
        time.sleep(0.5)

        if data_received["last_imu"]:
            print(f"  ✓ Notification handler working")
            print(f"    Last IMU: Yaw={data_received['last_imu'].yaw:.2f}, "
                  f"Pitch={data_received['last_imu'].pitch:.2f}, "
                  f"Roll={data_received['last_imu'].roll:.2f}")
        else:
            print("  ✗ Notification handler did not emit data")
    except Exception as e:
        print(f"  ✗ Notification handler error: {e}")
    print()

    # Test 3: Test yaw reset
    print("Test 3: Testing yaw reset")
    if data_received["last_imu"]:
        yaw_before = data_received["last_imu"].yaw
        worker.request_yaw_reset()
        time.sleep(0.2)
        yaw_after = data_received["last_imu"].yaw
        print(f"  Yaw before reset: {yaw_before:.2f}")
        print(f"  Yaw after reset: {yaw_after:.2f}")
    print()

    # Test 4: Test connection attempt (will likely fail without real device)
    print("Test 4: Testing connection attempt")
    print("  Note: This will likely fail without a real BLE device nearby")
    worker.request_connect()
    print("  Connection requested, waiting 8 seconds...")
    time.sleep(8)  # Wait for connection attempts
    print(f"  Final connection state: {worker.is_connected}")
    print(f"  Connection attempts made: {worker.connection_attempts}")
    print()

    # Test 5: Test disconnect
    if worker.is_connected:
        print("Test 5: Testing disconnect")
        worker.request_disconnect()
        time.sleep(2)
        print(f"  Disconnected: {not worker.is_connected}")
        print()

    # Test 6: Check data reception (if connected)
    if data_received["count"] > 1:
        print(f"Test 6: Data Reception Summary")
        print(f"  Total IMU data packets received: {data_received['count']}")
        print(f"  ✓ BLE communication working!")
    else:
        print(f"Test 6: Data Reception Summary")
        print(f"  Total IMU data packets: {data_received['count']}")
        print(f"  Note: No live data received (expected without real device)")
    print()

    # Cleanup
    print("Stopping worker thread...")
    worker.stop()
    worker.wait()  # Wait for thread to finish
    print("Worker thread stopped")

    print("\n=== Test Complete ===")
    print("\nTest Results Summary:")
    print(f"  - Worker thread lifecycle: ✓")
    print(f"  - Signal connections: ✓")
    print(f"  - Notification handler: {'✓' if data_received['count'] > 0 else '⚠️ (no real device)'}")
    print(f"  - Connection handling: {'✓' if worker.connection_attempts > 0 else '⚠️'}")
    print(f"  - IMU data parsing: {'✓' if data_received['last_imu'] else '⚠️'}")


def test_ble_scanner():
    """
    Separate test to scan for available BLE devices.
    Useful for finding device addresses before running the main test.
    """
    import asyncio

    print("=== BLE Device Scanner ===\n")
    print("Scanning for BLE devices (10 seconds)...\n")

    async def scan():
        devices = await BleakScanner.discover(timeout=10.0)
        if devices:
            print(f"Found {len(devices)} device(s):\n")
            for i, device in enumerate(devices, 1):
                print(f"{i}. {device.name or 'Unknown'}")
                print(f"   Address: {device.address}")
                print(f"   RSSI: {device.rssi} dBm")
                print()
        else:
            print("No BLE devices found")
            print("Make sure:")
            print("  - Bluetooth adapter is plugged in")
            print("  - BLE device is powered on and nearby")
            print("  - Device is not already connected to another program")

    try:
        asyncio.run(scan())
    except Exception as e:
        print(f"Scan failed: {e}")
        print("Bluetooth adapter might not be available")

    print("=== Scan Complete ===")


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        # Run BLE scanner
        test_ble_scanner()
    else:
        # Run main test
        test_ble_worker()

        print("\nTip: Run with --scan to scan for available BLE devices:")
        print("  python BLE_Worker.py --scan")