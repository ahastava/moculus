import asyncio
from bleak import  BleakClient, BleakScanner

async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        print(device)
        # print(f"Device Name: {device.name}")
        # print(f"Device Address: {device.address}")
        # print(f"Device UUIDs: {device.metadata['uuids']}")
        # print("-" * 40)

asyncio.run(main())
