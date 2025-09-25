import mysql_interface

data = mysql_interface.db_select(None, 'select * from probe_imu', [])
if len(data) == 0:
    mysql_interface.db_insert(None, 'insert into probe_imu(counter, roll, pitch, yaw) Value(%s,%s,%s,%s);', [1, 0, 0, 0])

print(data)