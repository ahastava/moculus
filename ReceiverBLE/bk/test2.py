import asyncio
import struct
from bleak import BleakClient

# Device and characteristic UUIDs
DEVICE_ADDRESS = "82:EC:2D:57:38:16"
CHARACTERISTIC_UUID_COUNTER = "00000000-0000-0000-0000-000000123400"
CHARACTERISTIC_UUID_YAW = "00000000-0000-0000-0000-0000001234aa"
CHARACTERISTIC_UUID_PITCH = "00000000-0000-0000-0000-0000001234bb"
CHARACTERISTIC_UUID_ROLL = "00000000-0000-0000-0000-0000001234cc"


# Dictionary to store grouped data
imu_data = {"counter": None, "yaw": None, "pitch": None, "roll": None}

# Notification handler for Yaw
def notification_handler_counter(sender, data):
    global imu_data
    imu_data["counter"] = int.from_bytes(data, byteorder='little')
    process_data()  # Check if all data is received

# Notification handler for Yaw
def notification_handler_yaw(sender, data):
    global imu_data
    imu_data["yaw"] = struct.unpack('<f', data)[0]
    process_data()  # Check if all data is received

# Notification handler for Yaw
def notification_handler_yaw(sender, data):
    global imu_data
    imu_data["yaw"] = struct.unpack('<f', data)[0]
    process_data()  # Check if all data is received

# Notification handler for Pitch
def notification_handler_pitch(sender, data):
    global imu_data
    imu_data["pitch"] = struct.unpack('<f', data)[0]
    process_data()  # Check if all data is received

# Notification handler for Roll
def notification_handler_roll(sender, data):
    global imu_data
    imu_data["roll"] = struct.unpack('<f', data)[0]
    process_data()  # Check if all data is received

# Function to process grouped data
def process_data():
    """Check if all data points (yaw, pitch, roll) are received, then process."""
    if None not in imu_data.values():  # Ensure all values are received
        print(f"Grouped Data: Counter = {imu_data['counter']} Yaw={imu_data['yaw']}, Pitch={imu_data['pitch']}, Roll={imu_data['roll']}")
        imu_data["counter"],imu_data["yaw"], imu_data["pitch"], imu_data["roll"] = None, None, None, None  # Reset for next cycle

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        if not client.is_connected:
            print("Failed to connect to the device.")
            return

        print("Connected to the device.")

        # Start notifications for each characteristic

        await client.start_notify(CHARACTERISTIC_UUID_COUNTER, notification_handler_counter)
        await client.start_notify(CHARACTERISTIC_UUID_YAW, notification_handler_yaw)
        await client.start_notify(CHARACTERISTIC_UUID_PITCH, notification_handler_pitch)
        await client.start_notify(CHARACTERISTIC_UUID_ROLL, notification_handler_roll)
        print("Started notifications. Waiting for updates...")

        # Keep script running to receive notifications
        while True:
            await asyncio.sleep(1)  # Sleep in small intervals to keep the event loop alive

        # Stop notifications
        await client.start_notify(CHARACTERISTIC_UUID_COUNTER)
        await client.stop_notify(CHARACTERISTIC_UUID_YAW)
        await client.stop_notify(CHARACTERISTIC_UUID_PITCH)
        await client.stop_notify(CHARACTERISTIC_UUID_ROLL)
        print("Stopped notifications.")

# Run the async function
asyncio.run(main())