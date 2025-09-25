import asyncio
from bleak import BleakClient
import aiomysql
from datetime import datetime
import logging
from collections import deque
import threading
import queue
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123123',
    'db': 'ble_receiver',
    'charset': 'utf8mb4',
    'autocommit': True
}

# BLE configuration
DEVICE_ADDRESS = "56:94:F5:36:EC:7E"
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

    async def update_current(self, counter, roll, pitch, yaw, timestamp):
        """Fast update of current record only"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        '''UPDATE probe_imu SET
                           counter = %s, roll = %s, pitch = %s, yaw = %s, updated_at = %s
                           WHERE idx = 1''',
                        (counter, roll, pitch, yaw, timestamp)
                    )
        except Exception as e:
            logging.error(f"Failed to update current record: {e}")

    def add_to_history_queue(self, counter, roll, pitch, yaw, timestamp):
        """Add to history queue (non-blocking)"""
        try:
            self.history_queue.put_nowait((counter, roll, pitch, yaw, timestamp))
        except queue.Full:
            logging.warning("History queue full, dropping oldest record")
            try:
                self.history_queue.get_nowait()  # Remove oldest
                self.history_queue.put_nowait((counter, roll, pitch, yaw, timestamp))
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
                        cursor.executemany(
                            '''INSERT INTO probe_imu_history(counter, roll, pitch, yaw, updated_at)
                               VALUES(%s, %s, %s, %s, %s)''',
                            batch
                        )
                        logging.debug(f"Inserted {len(batch)} history records")
                        batch.clear()
                        last_insert = current_time

                except queue.Empty:
                    # Timeout - insert any pending batch
                    if batch:
                        cursor.executemany(
                            '''INSERT INTO probe_imu_history(counter, roll, pitch, yaw, updated_at)
                               VALUES(%s, %s, %s, %s, %s)''',
                            batch
                        )
                        logging.debug(f"Inserted final {len(batch)} history records")
                        batch.clear()
                        last_insert = time.time()
                    continue

        except Exception as e:
            logging.error(f"History worker error: {e}")
        finally:
            if conn:
                conn.close()


# Global database manager
db_manager = None


def notification_handler_1(sender, data):
    """Ultra-fast notification handler"""
    try:
        text = data.decode('utf-8')
        parts = text.split(',')

        counter = int(parts[0])
        yaw = float(parts[1])
        pitch = float(parts[2])
        roll = float(parts[3])
        timestamp = datetime.now()

        print(f"Data: counter {counter}, yaw {yaw:.2f}, pitch {pitch:.2f}, roll {roll:.2f}")

        if db_manager:
            # Fast async update of current record
            asyncio.create_task(
                db_manager.update_current(counter, roll, pitch, yaw, timestamp)
            )

            # Queue for background history insert
            db_manager.add_to_history_queue(counter, roll, pitch, yaw, timestamp)

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