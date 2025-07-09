import serial
from serial import Serial
from PIL import Image
import numpy as np
from time import sleep
import serial.tools.list_ports
import re
import time

@staticmethod
def get_COM_port():
    serial_port = 'COM5'

    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description:
            serial_port = p.description
            serial_port = serial_port[serial_port.find("(")+1:serial_port.find(")")]

    ser = serial.Serial(serial_port, 38400, timeout=None)
    return ser

ser = get_COM_port()

class DataTransmission(object):

    def __init__(self) -> None:
        
        pass
    gyro_x_bg_total, gyro_y_bg_total, gyro_z_bg_total = 0, 0, 0 # These values keep a running log of the background increments (the "delta" in ambulance (pitch,yaw,roll) values). 
    # ^ It might be the case that we need to take initial readings and send those rather than incremental change values from 
    #   the amulance movement IMU, but I don't think it matters either way. Maybe at some point investigate this.

    baseline_gyro_x, baseline_gyro_y, baseline_gyro_z = 0, 0, 0    

    @staticmethod
    def provide_initial_prompt():
        print('\n\n\n\n\n')
        user_input = input('Please orient the probe approximately perpendicular to the surface of the mannequin\'s chest.\nYou will need to keep it in this position for approximately 30 seconds to allow the probe to be "zeroed-out".\nThe GUI will then appear and you will be free to move the probe around.\nPress the "Enter" key when ready:')

    @staticmethod
    def calibration_countdown(num_secs):
        print('GATHERING INITIAL PROBE MEASUREMENTS...')  # pi
        initial_time = time.time()
        current_time = initial_time
        z_coord, x_coord, y_coord, gyro_x, gyro_y, gyro_z =  None, None, None, None, None, None
        while (current_time-initial_time) < num_secs:
            try:
                z_coord, x_coord, y_coord, gyro_x, gyro_y, gyro_z = DataTransmission.get_mat_x_y_z__gyroX_gyroY_gyroZ(baseline_reading=True)
                current_time = time.time()
            finally:
                if((current_time-initial_time) >= num_secs):
                    DataTransmission.baseline_gyro_x = gyro_x
                    DataTransmission.baseline_gyro_y = gyro_y
                    DataTransmission.baseline_gyro_z = gyro_z


    @staticmethod
    def clean_up_probe_line(line):
        while(line[1:].startswith("BB")):
                line = line[1:]

        if line.startswith("BB") and "BB" in line[2:]:
            index_of_second_BB = line[2:].find("BB")
            line = line[0:index_of_second_BB+2] + "Y"  
            
        return line

    @staticmethod
    def find_decimals(string):
        pattern = r"^BB(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)Y$"
        match = re.match(pattern, string)
        if match:
            return tuple(float(x) for x in match.groups())
        return None

    @staticmethod
    def get_mat_x_y_z__gyroX_gyroY_gyroZ(baseline_reading=None):
        gyro_x, gyro_y, gyro_z = -1, -1, -1
        gyro_x_bg, gyro_y_bg, gyro_z_bg = -1, -1, -1 
        mat_x, mat_y, voltage = -1, -1, -1

        while True:
            try:
                #print('WAITING ', ser.in_waiting)
                if ser.in_waiting > 15:  # If there is serial data available
                    line = ser.readline()
                    line = line.decode("ANSI") #ser.readline returns a binary, convert to string
                    print('LINE AT BASELINE: ', line)
                    if(len(line) >= 6 and line.startswith("Calibrating")==False and line.startswith("state:")==False and line.startswith("AA")==False and line.startswith("BB")==False and line is not None):
                        voltage, mat_x, mat_y = line.split(" ")
                        mat_y = mat_y.split()[0]
                        voltage, mat_x, mat_y = int(voltage), int(mat_x), int(mat_y)

                    if((len(line) > 14  and line.startswith("state:_BB")==True or line.startswith("BB")==True)): #and line.endswith("ZZ")==True):
                        line=line.strip()
                        print('CLEANED UP PROBE LINE: ', line)
                        gyro_x, gyro_y, gyro_z = DataTransmission.find_decimals(line)                    
                        if(gyro_x != -1 and gyro_y != -1  and gyro_z != -1 and gyro_x_bg != -1 and gyro_y_bg != -1 and gyro_z_bg != -1):
                            break
                        
                    elif( (len(line) > 20 and line[7:].startswith("AA")==True) or (len(line) > 14 and line.startswith("AA")==True)):# background noise subtraction
                        line_bg_substring  = line[10:(len(line)-4)]
                        gyro_x_bg, gyro_y_bg, gyro_z_bg = line_bg_substring.split("_")
                        gyro_z_bg = gyro_z_bg.split()[0]
                        gyro_x_bg,gyro_y_bg,gyro_z_bg= float(gyro_x_bg.strip()), float(gyro_y_bg.strip()),float(gyro_z_bg.strip())
                        print('BACKGROUND LINE: ', line_bg_substring)
                        if(gyro_x != -1 and gyro_y != -1  and gyro_z != -1 and gyro_x_bg != -1 and gyro_y_bg != -1 and gyro_z_bg != -1 and voltage != -1 and mat_x != -1 and mat_y != -1):
                            break
          
            except Exception as d:
                print('ERROR: ', d)
                pass

        gyro_x = float(gyro_x)
        gyro_y = float(gyro_y)
        gyro_z = float(gyro_z)

        Serial.flush(ser)

        if(baseline_reading !=None):  # If we're taking the baseline reading (the calibration baseline: "zeroing" out at the start)
            return voltage, mat_x, mat_y, gyro_x-gyro_x_bg, gyro_y-gyro_y_bg, gyro_z-gyro_z_bg
        else:
            return voltage, mat_x, mat_y, gyro_x-gyro_x_bg-DataTransmission.baseline_gyro_x, gyro_y-gyro_y_bg-DataTransmission.baseline_gyro_y, gyro_z-gyro_z_bg-DataTransmission.baseline_gyro_z
        
