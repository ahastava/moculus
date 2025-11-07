import asyncio
from bleak import BleakClient
import aiomysql
from datetime import datetime
import logging
from collections import deque
import threading
import queue
import time

import config

# Database configuration
DB_CONFIG = config.DB_CONFIG

# BLE configuration for both devices
PROBE_ADDRESS = "BB:13:4D:D8:C7:42"  # First IMU
CAR_ADDRESS = "B5:78:F1:65:DA:E9"  # Second IMU - UPDATE THIS
CHARACTERISTIC_UUID_1 = "00000000-0000-0000-0000-0000001234DD"


class UltraFastDatabaseManager:
    def __init__(self, db_config, table_name, history_table_name):
        self.db_config = db_config
        self.table_name = table_name
        self.history_table_name = history_table_name
        self.pool = None
        self.shutdown_event = threading.Event()


    async def create_pool(self):
        """Create connection pool"""
        self.pool = await aiomysql.create_pool(
            minsize=2,
            maxsize=5,
            **self.db_config
        )

    async def close_pool(self):
        """Close connection pool"""
        self.shutdown_event.set()

        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logging.info(f"Database manager closed for {self.table_name}")

    async def update_current(self, data_dict, timestamp):
        """Fast update of current record with all fields"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        f'''UPDATE {self.table_name} SET
                           ble_counter = %s,
                           report_second = %s,
                           angle_report_loop_count = %s,
                           transferred_loop_count = %s,
                           status = %s,
                           roll = %s,
                           pitch = %s,
                           yaw = %s,
                           quat_r = %s,
                           quat_i = %s,
                           quat_j = %s,
                           quat_k = %s,
                           updated_at = %s
                           WHERE idx = 1''',
                        (
                            data_dict['ble_counter'],
                            data_dict['report_second'],
                            data_dict['angle_report_loop_count'],
                            data_dict['transferred_loop_count'],
                            data_dict['status'],
                            data_dict['roll'],
                            data_dict['pitch'],
                            data_dict['yaw'],
                            data_dict['quat_r'],
                            data_dict['quat_i'],
                            data_dict['quat_j'],
                            data_dict['quat_k'],
                            timestamp
                        )
                    )

                    # === 2️⃣ Insert into history table ===
                    await cursor.execute(
                        f'''INSERT INTO {self.history_table_name}
                                                  (ble_counter, report_second, angle_report_loop_count,
                                                   transferred_loop_count, status, roll, pitch, yaw,
                                                   quat_r, quat_i, quat_j, quat_k, updated_at)
                                               VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                        (
                            data_dict['ble_counter'],
                            data_dict['report_second'],
                            data_dict['angle_report_loop_count'],
                            data_dict['transferred_loop_count'],
                            data_dict['status'],
                            data_dict['roll'],
                            data_dict['pitch'],
                            data_dict['yaw'],
                            data_dict['quat_r'],
                            data_dict['quat_i'],
                            data_dict['quat_j'],
                            data_dict['quat_k'],
                            timestamp
                        )
                    )

                await conn.commit()
        except Exception as e:
            logging.error(f"Failed to update current record in {self.table_name}: {e}")


# Global database managers
probe_db_manager = None
car_db_manager = None


def create_notification_handler(db_manager, device_name):
    """Factory function to create notification handler for specific device"""

    def notification_handler(sender, data):
        """Ultra-fast notification handler"""
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

                quat_r = float(parts[8].replace("Q=", ""))
                quat_i = float(parts[9])
                quat_j = float(parts[10])
                quat_k = float(parts[11])

                # Parse all fields
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

                print(
                    f"[{device_name}] loop={BLECounter}, {reportSecond}, {angleReportLoopCount}, {transferredLoopCount} "
                    f"status={status}, "
                    f"YPR=({yaw:.2f}, {pitch:.2f}, {roll:.2f}), "
                    f"Quat=({quat_r:.3f}, {quat_i:.3f}, {quat_j:.3f}, {quat_k:.3f})"
                )

                if db_manager:
                    # Fast async update of current record
                    asyncio.create_task(
                        db_manager.update_current(data_dict, timestamp)
                    )

        except Exception as e:
            logging.error(f"[{device_name}] Notification error: {e}")

    return notification_handler


async def handle_device(device_address, db_manager, device_name):
    """Handle connection and notifications for a single device"""
    try:
        async with BleakClient(device_address) as client:
            if not client.is_connected:
                print(f"[{device_name}] Failed to connect to the device.")
                return

            print(f"[{device_name}] Connected to the device.")

            handler = create_notification_handler(db_manager, device_name)

            try:
                await client.start_notify(CHARACTERISTIC_UUID_1, handler)
                print(f"[{device_name}] Started notifications. Waiting for updates...")

                while True:
                    await asyncio.sleep(0.1)

            except (asyncio.CancelledError, KeyboardInterrupt):
                print(f"[{device_name}] Shutting down...")
                raise

            finally:
                if client.is_connected:
                    await client.stop_notify(CHARACTERISTIC_UUID_1)
                    await client.disconnect()
                    print(f"[{device_name}] Client disconnected.")

    except Exception as e:
        logging.error(f"[{device_name}] Device handler error: {e}")
        raise


async def main():
    global probe_db_manager, car_db_manager

    logging.basicConfig(level=logging.INFO)

    # Initialize database managers for both devices
    probe_db_manager = UltraFastDatabaseManager(DB_CONFIG, "probe_imu", "probe_imu_history")
    car_db_manager = UltraFastDatabaseManager(DB_CONFIG, "car_imu", "car_imu_history")

    await probe_db_manager.create_pool()
    await car_db_manager.create_pool()

    try:
        # Run both device handlers concurrently
        await asyncio.gather(
            handle_device(PROBE_ADDRESS, probe_db_manager, "PROBE"),
            handle_device(CAR_ADDRESS, car_db_manager, "CAR"),
            return_exceptions=True
        )

    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Main shutting down...")

    finally:
        if probe_db_manager:
            await probe_db_manager.close_pool()
        if car_db_manager:
            await car_db_manager.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
