import mysql.connector # mysql-connector-python

DB_HOST = 'localhost'
DB_PASS = '123123'
DB_NAME = 'ble_receiver'  # 'test'
DB_USER = 'root'


def db_raw_select(sql):
    con = mysql.connector.connect(host=DB_HOST,
                                  user=DB_USER,
                                  passwd=DB_PASS,
                                  db=DB_NAME,
                                  auth_plugin='mysql_native_password')
    cursor = con.cursor(dictionary=True)
    cursor.execute(sql)
    result = cursor.fetchall()
    con.close()
    return result


def db_select(logger, sql, data):
    try:
        con = mysql.connector.connect(host=DB_HOST,
                                      user=DB_USER,
                                      passwd=DB_PASS,
                                      db=DB_NAME,
                                      auth_plugin='mysql_native_password')
        cursor = con.cursor(dictionary=True)
        cursor.execute(sql, data)
        result = cursor.fetchall()
        con.close()
        return result
    except Exception as e:
        print(e)
        return e
    finally:
        if cursor:
            cursor.close()
        if con:
            con.close()

def db_insert(logger, sql, data):
    try:
        con = mysql.connector.connect(host=DB_HOST,
                                      user=DB_USER,
                                      passwd=DB_PASS,
                                      db=DB_NAME,
                                      auth_plugin='mysql_native_password')
        cursor = con.cursor(dictionary=True)
        cursor.execute(sql, data)
        con.commit()
        con.close()

    except Exception as e:
        print(e)
        return e
    finally:
        if cursor:
            cursor.close()
        if con:
            con.close()

# #db_raw_select('select 1')
# from datetime import datetime
# data = [1.1,2.2, 3.3, datetime.now()]
# print( datetime.now())
# db_insert(None, 'insert into angle(roll, pitch, yaw, dt2) Values(%s, %s, %s, %s)', data) # 0.02s
# print( datetime.now())

