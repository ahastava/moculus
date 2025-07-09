import sys
#import get_data
#import transmit_data
#from transmit_data import DataTransmission
#import serial
import time 

from PyQt5 import QtCore, QtGui, QtWidgets, Qt

#ser = transmit_data.DataTransmission.get_COM_port()
#ser = serial.Serial(serial_port, 38400, timeout=None)

US_IMAGE_HEIGHT = 720
US_IMAGE_WIDTH = 1280

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.title = 'Ultrasound GUI'
        self.left = 10
        self.top = 10
        self.total_frames = 0
        self.total_time = 0
        self.initUI()
        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z = 0
        self.x_coord = 0
        self.y_coord = 0
        self.voltage = 0

    def set_background_image(self, image_name):
        background = QtGui.QPixmap(image_name)
        brush = QtGui.QBrush(background)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setBrush(QtGui.QPalette.Window, brush)
        self.setPalette(p)

    def initUI(self):
        self.mdic_label = QtWidgets.QLabel(self)
        self.mdic_label.setPixmap(QtGui.QPixmap("MDIC_logo.png"))
        self.mdic_label.setAlignment(QtCore.Qt.AlignCenter)

        scaled_mdic_pixmap = self.mdic_label.pixmap().scaled(int(self.mdic_label.pixmap().width()*0.7),int(self.mdic_label.pixmap().height()*0.55))
        self.mdic_label.setPixmap(scaled_mdic_pixmap)
        self.mdic_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
       

        self.layout = QtWidgets.QVBoxLayout(self)

        # Add the mdic logo image label to the top of the layout
        self.layout.addWidget(self.mdic_label)
        self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
 
        self.layout.addWidget(self.label)
        self.layout.setContentsMargins(100,0,600,200)  # Add some margins (extra on the right to accomodate the gyro_readings text section)
        self.set_background_image("pastel-blue-vignette-concrete-textured-background.png")  # set the background image for the entire app
        
        self.initiate_gyro_readings_text_section()  # adds the text section for the gyro readings
        self.initiate_torso_images_section()

        # Run an initial 10 second calibration. 
#        DataTransmission.provide_initial_prompt()
#        DataTransmission.calibration_countdown(5)  # initial 15 second countdown (right now I suspect it needs about 15 seconds to zero-out)

        # create a timer for updating images.
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start()  # sets timer to take a new reading every 150 ms
        self.update_image()

    def initiate_torso_images_section(self):
        path = 'body_torso_small_color_1.png'
        self.torso_pixmap = QtGui.QPixmap(path)
        self.torso_label = QtWidgets.QLabel(self)
        self.torso_label.setAlignment(QtCore.Qt.AlignLeft)
        self.torso_label.setFixedSize(320, 380)

        self.draw_pixmap = QtGui.QPixmap(self.torso_pixmap.size())
        self.draw_pixmap.fill(QtCore.Qt.transparent)

        self.update_torso_image(0, 0)  # Initially draw the points

        self.torso_label.setPixmap(self.torso_pixmap)
        self.torso_label.move(self.label.x() + US_IMAGE_WIDTH + 150, self.label.y() + int(US_IMAGE_HEIGHT / 2) + 230)


    def update_torso_image(self, x_coord, y_coord):
        """
        This function updates the dot overlapping the torso image on the right-hand side of the torso image. 
        Depending on where the user is pressing, the dot's location changes.          
        """

        merged_pixmap = self.torso_pixmap.copy()  # Create a copy of the original pixmap

        painter = QtGui.QPainter(merged_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Clear the draw_pixmap area by filling it with the torso_pixmap
        painter.drawPixmap(0, 0, self.torso_pixmap)

        pen = QtGui.QPen()
        pen.setWidth(10)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)

        # Calculate the center coordinates of the dot
        dot_radius = 5  # 5 pixel radius

        # Determine the location of the dot based on some arbitrary approximation 
        # TODO determine the proper coordinate location based on the image w/ grid overlay provided by Thea. 
        dot_center = QtCore.QPoint(60 + x_coord * 13, 90 + y_coord * 17)  # Adjust the coordinates as needed

        # Draw a rounded dot where elipse radii are equal on x and y (i.e. make a goddamn circle)
        painter.drawEllipse(dot_center, dot_radius, dot_radius)

        painter.end()

        # Create a rounded mask (this will be to round the corner on the torso image label thing. )
        mask_radius = 30  # Adjust the radius as desired
        rounded_mask = QtGui.QPixmap(merged_pixmap.size())
        rounded_mask.fill(QtCore.Qt.transparent)

        mask_painter = QtGui.QPainter(rounded_mask)
        mask_painter.setRenderHint(QtGui.QPainter.Antialiasing)
        mask_painter.setBrush(QtCore.Qt.white)
        mask_painter.setPen(QtCore.Qt.NoPen)
        mask_painter.drawRoundedRect(0, 0, rounded_mask.width(), rounded_mask.height(), mask_radius, mask_radius)
        mask_painter.end()

        # Apply the mask to the merged pixmap
        merged_pixmap.setMask(rounded_mask.mask())

        self.torso_label.setPixmap(merged_pixmap)


    def initiate_gyro_readings_text_section(self):
        self.text_label = QtWidgets.QLabel(self)
        self.text_label.setFixedSize(360,300)
        # self.text_label.  # set palette later
        self.text_label.setAlignment(QtCore.Qt.AlignLeft)
        self.text_label.setStyleSheet("background-color: white;")        
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.white)
        self.text_label.setStyleSheet("QLabel { background-color: charcoal; "
                            "border-color: white; "
                            "border-style: outset;"
                            "border-width: 2px;"
                            "border-radius: 30px; }")        
        font = QtGui.QFont()
        font.setBold(True)
        self.text_label.setFont(font)
        self.text_label.setAutoFillBackground(True)
        self.text_label.setPalette(palette)
        #self.text_label.raise_()

        #self.layout.addWidget(self.text_label)

        self.text_label.move(self.label.x()+US_IMAGE_WIDTH+150, self.label.y()+int(US_IMAGE_HEIGHT/2)-110)


    def get_rotation_tilt_pressure_pic_string(self, rotation, tilt, pressure):
        rotation_list = ["NEUTRAL", "CW90", "CW180", "CW270"]
        tilt_list = ["NEUTRAL", "FORWARD45", "BACKWARD45"]  # forward 45  = if tilt is between -25 and -65 let's say. backwards 45 between 25 and 65
        pressure_list = ["L1", "L2", "L3"]

        r_string, t_string, p_string = "NEUTRAL", "NEUTRAL", "L1"

        #print ("BEGINNING ROTATION SUBSTRING GET")
        if rotation < -70 and rotation >=-110:
            r_string = rotation_list[1]
        elif rotation <-110 and rotation >= -250:
            r_string = rotation_list[2]
        elif rotation < -250 and rotation >= -290:
            r_string = rotation_list[3]
        else:
            r_string = rotation_list[0]

        #print ("BEGINNING TILT SUBSTRING GET")

        if (tilt <= 20 and tilt >-20):
            t_string = tilt_list[0]
        elif tilt <-25:
            t_string = tilt_list[1]
        elif tilt > 25: 
            t_string = tilt_list[2]
        
        #print ("BEGINNING PRESSURE SUBSTRING GET")

        if(pressure < 650):
            p_string = pressure_list[0]
        elif pressure < 800:
            p_string = pressure_list[1]
        else:
            p_string = pressure_list[2]

        #print('RSTRING, TSTRING, PSTRING = {0}, {1}, {2}'.format(r_string, t_string, p_string))
        return r_string + "__" + t_string + "__" + p_string
            

    def update_average_framerate(self, gyro_time):
        self.total_frames = self.total_frames + 1
        self.total_time = self.total_time + gyro_time


    def update_image(self):

        #  Record the start time for this cycle. We will also take the time after the coordinates are retrieved. 
        #  (we keep this to get a good esimate of the number of images per second)
        time1 = time.time()        

        #z_coord, x_coord, y_coord, gyro_x, gyro_y, gyro_z = transmit_data.DataTransmission.get_mat_x_y_z__gyroX_gyroY_gyroZ()
        z_coord, x_coord, y_coord, gyro_x, gyro_y, gyro_z = 1,2,3,10,11,12
        self.x_coord = x_coord; self.y_coord = y_coord; self.voltage = z_coord
        gyro_x = round(gyro_x, 2)
        gyro_z = round(gyro_z, 2)
        self.gyro_x = gyro_x
        self.gyro_z = gyro_z
        time2 = time.time()

        self.gyro_time = time2-time1
        #print('GYRO  TIME = ', self.gyro_time, " seconds." )
        self.update_average_framerate(self.gyro_time)
        self.update_sensor_gui_values()

        self.update_torso_image(self.x_coord, self.y_coord)

        if(z_coord != 0 and x_coord != (0 or None) and y_coord != (0 or None)):  # If the values are valid
            print('\n\n')
            print("(VOLTAGE, X, Y) = ", z_coord, x_coord, y_coord)
            print("(YAW, ROLL) = ",gyro_x, "°, ", gyro_z, "°")
            try:
                rotation_pic_mapping = self.get_rotation_tilt_pressure_pic_string(gyro_x, gyro_z, z_coord)
                path = 'ultrasound_images/x_' + str(x_coord) + '/y_' + str(y_coord) + '/{0}.jpg'.format(rotation_pic_mapping)
                print('path: ', path)
                pixmap = QtGui.QPixmap(path)

                if not pixmap.isNull():  # If the 
                    self.scaled_pixmap = pixmap.scaled(US_IMAGE_WIDTH, US_IMAGE_HEIGHT)
                    self.label.setPixmap(self.scaled_pixmap)
                    self.label.adjustSize()

                    #self.resize(scaled_pixmap.size())
                    self.label.setAlignment(QtCore.Qt.AlignCenter)
                    self.update_sensor_gui_values()

            except Exception:
                print(Exception)
                return
        else:
            path = 'ultrasound_images/x_' + str(7) + '/y_' + str(7) + '/{0}.jpg'.format('test1')
            pixmap = QtGui.QPixmap(path)
            self.scaled_pixmap = pixmap.scaled(US_IMAGE_WIDTH, US_IMAGE_HEIGHT)
            if self.scaled_pixmap.isNull:
                self.scaled_pixmap = QtGui.QPixmap(US_IMAGE_WIDTH, US_IMAGE_HEIGHT)
                self.scaled_pixmap.fill(QtGui.QColor("charcoal"))
            self.label.setPixmap(self.scaled_pixmap)
            self.label.adjustSize()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            return
                
    def update_sensor_gui_values(self):

        self.text_label.setText("                    PROBE VALUES\n_____________________________________" +
                                "\n\n   YAW: {0}°\n   ROLL: {1}° \n_____________________________________\n\n  Current Frame Rate: {2}\n  Average Frame Rate: {3} \n\n   (X,Y) = ({4}, {5})\n\n_____________________________________\nVoltage (∝ pressure) = {6}".format(self.gyro_x, self.gyro_z, (round(1./self.gyro_time, 1) if self.gyro_time != 0 else "N/A"), (round(self.total_frames/self.total_time, 1) if self.total_time !=0 else "N/A"), self.x_coord, self.y_coord, self.voltage))
    





if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())


