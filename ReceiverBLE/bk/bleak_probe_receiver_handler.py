import asyncio
from bleak import BleakClient

# Replace with your BLE device address and characteristic UUID
DEVICE_ADDRESS = "BB:13:4D:D8:C7:42" #"82:EC:2D:57:38:16"
CHARACTERISTIC_UUID = "00000000-0000-0000-0000-000000123400"
CHARACTERISTIC_UUID_1 = "00000000-0000-0000-0000-0000001234DD"

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

   # print(f"Notification from :", data)
    text = data.decode('utf-8')
    parts = text.split(',')

    counter = int(parts[0])
    yaw = float(parts[1])
    pitch = float(parts[2])
    roll = float(parts[3])

    print(f"Data received: counter {counter}, yaw {yaw}, pitch: {pitch}, roll {roll}")

    #return

    # data = [counter, roll, pitch, yaw, datetime.now()]
    # # print( datetime.now())
    # mysql_interface.db_insert(None,
    #                           '''
    #                           update probe_imu
    #                           set
    #                               counter = %s,
    #                               roll = %s,
    #                               pitch = %s,
    #                               yaw = %s,
    #                               updated_at = %s
    #                           where idx = 1
    #                           ''',
    #                           data)  # 0.02s

    # mysql_interface.db_insert(None, '''
    #                                        insert into probe_imu_history(counter, roll, pitch, yaw, updated_at)
    #                                                    Values(%s, %s, %s, %s, %s)''', data)  # 0.02s

   # yaw_float = data.unpack('<f', data)[0]
   # yaw_float = struct.unpack('<f', data)[0]
   # value = int.from_bytes(data, byteorder='little')  # Assuming the data is an integer
   # print(f"Notification from {sender}: {yaw_float}")

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        # Ensure the device is connected
        if not client.is_connected:
            print("Failed to connect to the device.")
            return

        print("Connected to the device.")

        # Start notifications
        # await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        # print("Started notifications. Waiting for updates...")

        try:
            await client.start_notify(CHARACTERISTIC_UUID_1, notification_handler_1)
            print("Started notifications 1. Waiting for updates...")

            # Need this code, otherwise the program will exit
            # Keep the script running to receive notifications
            while True:
                await asyncio.sleep(0.1) # sleep for 1 second
        except asyncio.CancelledError:
            print("Cancelled by user or IDE stop button.")
        except KeyboardInterrupt:
            print("Interrupted by user.")

        finally:
            if client.is_connected:
                print("Stopping connection.")

                try:
                  #  await client.stop_notify(CHARACTERISTIC_UUID)   #the following is useless since it is already stoped nofity
                    await client.disconnect()
                    print("Client disconnected.")
                except Exception as e:
                    print("Error during disconnect:", e)
            else:
                print("Client already disconnected, exiting immediately.")



# Run the async function
asyncio.run(main())