import mysql.connector # mysql-connector-python
import config
from PyQt6.QtCore import QThread, pyqtSignal

DB_HOST = config.DB_CONFIG['host']
DB_PASS = config.DB_CONFIG['password']
DB_NAME = config.DB_CONFIG['db']
DB_USER = config.DB_CONFIG['user']


class DBWorker(QThread):

    error_signal = pyqtSignal(str)

    def db_raw_select(self, sql):
        try:
            con = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                passwd=DB_PASS,
                db=DB_NAME,
                auth_plugin='mysql_native_password'
            )
            cursor = con.cursor(dictionary=True)

            cursor.execute(sql)

            # If it's a SELECT, fetch results
            if sql.strip().lower().startswith("select"):
                result = cursor.fetchall()
            else:
                # For DELETE/UPDATE/INSERT or SET commands, commit changes
                con.commit()
                result = f"OK ({cursor.rowcount} rows affected)"

            cursor.close()
            con.close()
            return result

        except Exception as e:
            print("DB error:", e)

            self.error_signal.emit(e)
            return e


def db_select(logger, sql, data):

    con = None
    cursor = None
    try:
        con = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            passwd=DB_PASS,
            db=DB_NAME,
            auth_plugin='mysql_native_password'
        )
        cursor = con.cursor(dictionary=True)
        cursor.execute(sql, data)
        result = cursor.fetchall()
        return result

    except Exception as e:
        if logger:
            logger.error(f"DB select failed: {e}")
        else:
            print(f"DB select failed: {e}")
        return None  # or return e if you want to propagate

    finally:
        if cursor is not None:
            cursor.close()
        if con is not None:
            con.close()

def db_insert(logger, sql, data):

    con = None
    cursor = None

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
# # db_insert(None, 'insert into angle(roll, pitch, yaw, dt2) Values(%s, %s, %s, %s)', data) # 0.02s
# data = [1, 1, 1, 1, datetime.now()]
# # print( datetime.now())
# db_insert(None,
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
#
# print( datetime.now())


#db_raw_select("SET SQL_SAFE_UPDATES = 0;")
# d = db_raw_select("DELETE FROM probe_imu_history;")
# d = db_raw_select("SET SQL_SAFE_UPDATES = 1;")

# a = db_raw_select('select * from probe_imu')
# print(a)
