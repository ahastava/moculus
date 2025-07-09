import log
import mysql.connector
import config

def db_insert(sql, data):
    try:
        con = mysql.connector.connect(host = config.DB_HOST, user = config.DB_USER, passwd = config.DB_PASS, db = config.DB_NAME)
        cursor = con.cursor()
        cursor.execute(sql, data)
        con.commit()
        cursor.close()
        con.close()
        return True
    except Exception as e:
        try:
            # Try to close the connection if it exists
            con.close()
        except NameError:
            # No connection to close
            return None

        logger = log._generate_log()
        logger.error(e)
        logger.info(sql)
        logger.info(data)
        return e


def db_select(sql, data):
    try:
        con = mysql.connector.connect(host = config.DB_HOST, user = config.DB_USER, passwd = config.DB_PASS, db = config.DB_NAME)
        cursor = con.cursor(dictionary=True)
        cursor.execute(sql, data)
        result = cursor.fetchall()
        con.close()
        return result
    except Exception as e:
        try:
            # Try to close the connection if it exists
            con.close()
        except NameError:
            # No connection to close
            return None
        print(e)
        logger = log._generate_log()
        logger.error(e)
        logger.info(sql)
        logger.info(data)
        return e

def db_dirty_select(sql):
    try:
        con = mysql.connector.connect(host = config.DB_HOST, user = config.DB_USER, passwd = config.DB_PASS, db = config.DB_NAME)
        cursor = con.cursor(dictionary=True)
        cursor.execute(sql)
        result = cursor.fetchall()
        con.close()
        return result
    except Exception as e:
        try:
            # Try to close the connection if it exists
            con.close()
        except NameError:
            # No connection to close
            return None
        print(e)
        logger = log._generate_log()
        logger.error(e)
        logger.info(sql)
        return e


a = db_dirty_select('select * from nlog123 limit 10')
for item in a:
    print(item)