import asyncio
from bleak import  BleakClient, BleakScanner

import struct
from datetime import datetime
import mysql_interface

###

async def main():
    # devices = await BleakScanner.discover()
    # for d in devices:
    #     print(d)

    data = mysql_interface.db_select(None, 'select * from probe_imu', [])
    if len(data) == 0:
        mysql_interface.db_insert(None, 'insert into probe_imu(counter, roll, pitch, yaw) Value(%s,%s,%s,%s);',
                                  [1, 0, 0, 0])

    mysql_interface.db_insert(None, 'TRUNCATE TABLE probe_imu_history', [])

    device = await BleakScanner.find_device_by_address(
        'BB:13:4D:D8:C7:42'
    )
    if device is None:
        print("could not find device with address ")
        return
    else:
        print( device)
        print(f"Device Name: {device.name}")
        print(f"Device Address: {device.address}")
        print(f"Device UUIDs: {device.metadata['uuids']}")

       # global x_data, y_data

        # Connect to the device
        async with BleakClient(device) as client:
            print("Connected to device")
            services = await client.get_services()
            print("Services:", services)

            for service in services:
                print(service)
                for char in service.characteristics:
                    print(char)


            CHARACTERISTIC_UUID_COUNTER = '00000000-0000-0000-0000-000000123400'
            CHARACTERISTIC_UUID_YAW     = '00000000-0000-0000-0000-0000001234aa'
            CHARACTERISTIC_UUID_PITCH  = '00000000-0000-0000-0000-0000001234bb'
            CHARACTERISTIC_UUID_ROLL  = '00000000-0000-0000-0000-0000001234cc'
            # Check if the characteristic exists
            if CHARACTERISTIC_UUID_COUNTER not in [char.uuid for service in services for char in service.characteristics]:
                print(f"Characteristic {CHARACTERISTIC_UUID_COUNTER} not found!")
                return

            if CHARACTERISTIC_UUID_YAW not in [char.uuid for service in services for char in service.characteristics]:
                print(f"Characteristic {CHARACTERISTIC_UUID_YAW} not found!")
                return

            if CHARACTERISTIC_UUID_PITCH not in [char.uuid for service in services for char in service.characteristics]:
                print(f"Characteristic {CHARACTERISTIC_UUID_PITCH} not found!")
                return

            if CHARACTERISTIC_UUID_ROLL not in [char.uuid for service in services for char in service.characteristics]:
                print(f"Characteristic {CHARACTERISTIC_UUID_ROLL} not found!")
                return

            # Continuously read and print the characteristic value
            try:
                while True:

                  #  print('1:', datetime.now())

                    counter = await client.read_gatt_char(CHARACTERISTIC_UUID_COUNTER)

                    # b'\xe1\x07\x00\x00' = 2017
                    # 0xE1 = 225(decimal)
                    # 0x07×256 = 7×256 = 1792
                    # 225+1792+0+0=2017
                    counter_int = int.from_bytes(counter, byteorder='little')
                 #   print(f"Characteristic Value Counter: {counter} {counter_int}")

                #    print('2:', datetime.now())

                    yaw = await client.read_gatt_char(CHARACTERISTIC_UUID_YAW)
                    yaw_float = struct.unpack('<f', yaw)[0]
                 #   print(f"Characteristic Value 2: {yaw} {yaw_float}") # 123456789.123456789123456789 => 123456792.0
                #    print('3:', datetime.now())

                    pitch = await client.read_gatt_char(CHARACTERISTIC_UUID_PITCH)
                    pitch_float = struct.unpack('<f', pitch)[0]
                   # print(f"Characteristic Value 3: {pitch} {pitch_float}") # 123456789.123456789123456789 => 123456792.0

                   # print('4:', datetime.now())

                    roll = await client.read_gatt_char(CHARACTERISTIC_UUID_ROLL)
                    roll_float = struct.unpack('<f', roll)[0]
               #     print(f"Characteristic Value 4: {roll} {roll_float}") # 123456789.123456789123456789 => 123456792.0

                    print(f"Characteristic Value: {counter_int} {yaw_float} {pitch_float} {roll_float}")
                  #  print('5:', datetime.now())
                    data = [counter_int, roll_float, pitch_float, yaw_float, datetime.now()]
                    # print( datetime.now())
                    mysql_interface.db_insert(None,
                                              '''
                                              update probe_imu 
                                              set 
                                                  counter = %s,
                                                  roll = %s,
                                                  pitch = %s,
                                                  yaw = %s,
                                                  updated_at = %s
                                              where idx = 1
                                              ''',
                                              data)  # 0.02s

                    mysql_interface.db_insert(None, '''
                                        insert into probe_imu_history(counter, roll, pitch, yaw, updated_at)
                                                    Values(%s, %s, %s, %s, %s)''', data) # 0.02s
                   # print('6:', datetime.now())
                    await asyncio.sleep(0.01)  # Adjust the delay as needed
            except asyncio.CancelledError:
                print("Stopped reading characteristic.")


# # Create animation
#ani = FuncAnimation(fig, update, frames=itertools.count(), interval=200, blit=True)

# Show the plot
#plt.show()

asyncio.run(main())

