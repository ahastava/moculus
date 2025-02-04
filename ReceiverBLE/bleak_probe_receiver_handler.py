import asyncio
from bleak import BleakClient
import struct

# Replace with your BLE device address and characteristic UUID
DEVICE_ADDRESS = "82:EC:2D:57:38:16"
CHARACTERISTIC_UUID = "00000000-0000-0000-0000-000000123400"
CHARACTERISTIC_UUID_1 = "00000000-0000-0000-0000-0000001234aa"

# Notification handler function
def notification_handler(sender, data):
    """
    Callback function to handle notifications.
    :param sender: The handle of the characteristic that sent the notification.
    :param data: The data received in the notification (bytes).
    """
    value = int.from_bytes(data, byteorder='little')  # Assuming the data is an integer
    print(f"Notification from {sender}: {value}")

# Notification handler function
def notification_handler_1(sender, data):
    """
    Callback function to handle notifications.
    :param sender: The handle of the characteristic that sent the notification.
    :param data: The data received in the notification (bytes).
    """
  #  yaw = await client.read_gatt_char(CHARACTERISTIC_UUID_YAW)
   # yaw_float = data.unpack('<f', data)[0]
    yaw_float = struct.unpack('<f', data)[0]
   # value = int.from_bytes(data, byteorder='little')  # Assuming the data is an integer
    print(f"Notification from {sender}: {yaw_float}")

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        # Ensure the device is connected
        if not client.is_connected:
            print("Failed to connect to the device.")
            return

        print("Connected to the device.")

        # Start notifications
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Started notifications. Waiting for updates...")

        await client.start_notify(CHARACTERISTIC_UUID_1, notification_handler_1)
        print("Started notifications 1. Waiting for updates...")

        # Keep the script running to receive notifications
        await asyncio.sleep(30)  # Adjust the duration as needed

        # Stop notifications
        await client.stop_notify(CHARACTERISTIC_UUID)
        print("Stopped notifications.")

# Run the async function
asyncio.run(main())