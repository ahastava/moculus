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

# BLE configuration
# test  BB:13:4D:D8:C7:42
# probe 56:94:F5:36:EC:7E
# box 1 B5:78:F1:65:DA:E9
DEVICE_ADDRESS = "B5:78:F1:65:DA:E9"  # B5:78:F1:65:DA:E9
CHARACTERISTIC_UUID_1 = "00000000-0000-0000-0000-0000001234DD"


class UltraFastDatabaseManager:
    def __init__(self, db_config):
        self.db_config = db_config
        self.pool = None
        self.history_queue = queue.Queue()
        self.history_thread = None
        self.shutdown_event = threading.Event()

    async def create_pool(self):
        """Create connection pool"""
        self.pool = await aiomysql.create_pool(
            minsize=2,
            maxsize=5,
            **self.db_config
        )

        # Start background thread for history inserts
        self.history_thread = threading.Thread(target=self._history_worker, daemon=True)
        self.history_thread.start()

        logging.info("Database manager initialized")

    async def close_pool(self):
        """Close connection pool"""
        self.shutdown_event.set()
        if self.history_thread:
            self.history_thread.join(timeout=5)

        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logging.info("Database manager closed")

    async def update_current(self, data_dict, timestamp):
        """Fast update of current record with all fields"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        '''UPDATE probe_imu SET
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
        except Exception as e:
            logging.error(f"Failed to update current record: {e}")

    def add_to_history_queue(self, data_dict, timestamp):
        """Add to history queue (non-blocking)"""
        data_dict['timestamp'] = timestamp
        try:
            self.history_queue.put_nowait(data_dict)
        except queue.Full:
            logging.warning("History queue full, dropping oldest record")
            try:
                self.history_queue.get_nowait()  # Remove oldest
                self.history_queue.put_nowait(data_dict)
            except queue.Empty:
                pass

    def _history_worker(self):
        """Background thread worker for history inserts"""
        import mysql.connector

        conn = None
        batch = []
        last_insert = time.time()

        try:
            # Create dedicated connection for history thread
            conn = mysql.connector.connect(**{
                'host': self.db_config['host'],
                'user': self.db_config['user'],
                'password': self.db_config['password'],
                'database': self.db_config['db'],
                'autocommit': True
            })
            cursor = conn.cursor()

            while not self.shutdown_event.is_set():
                try:
                    # Wait for data with timeout
                    data = self.history_queue.get(timeout=0.5)
                    batch.append(data)

                    # Insert batch when full or timeout reached
                    current_time = time.time()
                    if (len(batch) >= 20 or
                            current_time - last_insert >= 2.0):
                        self._insert_batch(cursor, batch)
                        batch.clear()
                        last_insert = current_time

                except queue.Empty:
                    # Timeout - insert any pending batch
                    if batch:
                        self._insert_batch(cursor, batch)
                        batch.clear()
                        last_insert = time.time()
                    continue

        except Exception as e:
            logging.error(f"History worker error: {e}")
        finally:
            if conn:
                conn.close()

    def _insert_batch(self, cursor, batch):
        """Insert batch of history records"""
        values = [
            (
                data['ble_counter'],
                data['report_second'],
                data['angle_report_loop_count'],
                data['transferred_loop_count'],
                data['status'],
                data['roll'],
                data['pitch'],
                data['yaw'],
                data['quat_r'],
                data['quat_i'],
                data['quat_j'],
                data['quat_k'],
                data['timestamp']
            )
            for data in batch
        ]
        cursor.executemany(
            '''INSERT INTO probe_imu_history
               (ble_counter, report_second, angle_report_loop_count, transferred_loop_count,
                status, roll, pitch, yaw, quat_r, quat_i, quat_j, quat_k,
                updated_at)
               VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
            values
        )
        logging.debug(f"Inserted {len(batch)} history records")


# Global database manager
db_manager = None


def notification_handler_1(sender, data):
    """Ultra-fast notification handler"""
    try:
        text = data.decode('utf-8').strip()
        parts = [p.strip() for p in text.split(',')]
        print(parts)

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

            # accel_x = float(parts[12].replace("A=", ""))
            # accel_y = float(parts[13])
            # accel_z = float(parts[14])

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
                f"loop={BLECounter}, {reportSecond}, {angleReportLoopCount}, {transferredLoopCount} "
                f"status={status}, "
                f"YPR=({yaw:.2f}, {pitch:.2f}, {roll:.2f}), "
                f"Quat=({quat_r:.3f}, {quat_i:.3f}, {quat_j:.3f}, {quat_k:.3f}), "
            #    f"Accel=({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f})"
            )

            if db_manager:
                # Fast async update of current record
                asyncio.create_task(
                    db_manager.update_current(data_dict, timestamp)
                )

                # Queue for background history insert
                db_manager.add_to_history_queue(data_dict, timestamp)

    except Exception as e:
        logging.error(f"Notification error: {e}")


async def main():
    global db_manager

    logging.basicConfig(level=logging.INFO)

    # Initialize database manager
    db_manager = UltraFastDatabaseManager(DB_CONFIG)
    await db_manager.create_pool()

    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            if not client.is_connected:
                print("Failed to connect to the device.")
                return

            print("Connected to the device.")

            try:
                await client.start_notify(CHARACTERISTIC_UUID_1, notification_handler_1)
                print("Started notifications. Waiting for updates...")

                while True:
                    await asyncio.sleep(0.1)

            except (asyncio.CancelledError, KeyboardInterrupt):
                print("Shutting down...")

            finally:
                if client.is_connected:
                    await client.stop_notify(CHARACTERISTIC_UUID_1)
                    await client.disconnect()
                    print("Client disconnected.")

    finally:
        if db_manager:
            await db_manager.close_pool()


if __name__ == "__main__":
    asyncio.run(main())