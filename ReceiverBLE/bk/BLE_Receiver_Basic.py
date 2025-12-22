import asyncio
from bleak import BleakClient

# Update this with your Arduino's BLE address
DEVICE_ADDRESS = "BB:13:4D:D8:C7:42" #"82:EC:2D:57:38:16"
CHAR_UUID = "00000000-0000-0000-0000-0000001234DD"  # lowercase for Windows

def notification_handler(sender, data):
    message = data.decode('utf-8')
    print(f"Received: {message}")


async def main():
    print(f"Connecting to {DEVICE_ADDRESS}...")

    async with BleakClient(DEVICE_ADDRESS, timeout=30.0) as client:
        print(f"Connected: {client.is_connected}")

        # Start notifications
        await client.start_notify(CHAR_UUID, notification_handler)
        print("Listening for notifications...")

        # Keep listening for 30 seconds
        await asyncio.sleep(60)

        # Stop notifications
        await client.stop_notify(CHAR_UUID)
        print("Stopped")


if __name__ == "__main__":
    asyncio.run(main())

