import sys
import numpy as np
import cupy as cp
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QStyle, QHBoxLayout, \
    QSpacerItem, QPushButton, QStackedLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
import time

from volume import Volume  # Import the Volume class

from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # Works in PyQt6
import OpenGL.GL as gl # install PyOpenGL

import mysql_interface
from OpenGL.GL import *
from OpenGL.GLU import *

class ArrowGLWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.shift_x = 0
        self.shift_y = 0

        try:
            self.background_texture = self.load_texture("../data/test1.jpg")
        except Exception as e:
            print(e)

    def load_texture(self, path):
        try:
            img = QImage(path)
            img = img.convertToFormat(QImage.Format.Format_RGBA8888)
            width = img.width()
            height = img.height()

            # Convert QImage to numpy array
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))

            arr = np.flip(arr, axis=0)
            # ptr = img.bits()
            # ptr.setsize(img.sizeInBytes())

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, arr)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        except Exception as e:
            print(e)

        return texture_id

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def draw_axis_arrow(self, axis, length, color):
        glPushMatrix()
        glColor3f(*color)

        # Rotate according to axis
        if axis == 'x':
            glRotatef(90, 0, 1, 0)
        elif axis == 'y':
            glRotatef(-90, 1, 0, 0)
        elif axis == 'z':
            pass  # default is +Z

        # Draw shaft
        quad = gluNewQuadric()
        gluCylinder(quad, 0.05, 0.05, length, 20, 1)

        # Draw arrow head (cone)
        glTranslatef(0, 0, length)
        gluCylinder(quad, 0.1, 0.0, 0.3, 20, 1)
        gluDeleteQuadric(quad)
        print("axis\n")
        glPopMatrix()

    def paintGL(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Disable depth for background
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.background_texture)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)

        glTexCoord2f(1, 0)
        glVertex2f(1, 0)

        glTexCoord2f(1, 1)
        glVertex2f(1, 1)

        glTexCoord2f(0, 1)
        glVertex2f(0, 1)
        glEnd()

        glDisable(GL_TEXTURE_2D)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)


        glLoadIdentity()
        gluLookAt(4, -4, 6, 0, 0, 0, 0, 0, 1)

        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # glOrtho(-5, 5, -5, 5, -10, 10)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        # # glRotatef(35, 1, 0, 0)  # simulate tilt (look down)
        # # glRotatef(-60, 0, 1, 0)  # simulate side angle



        #glOrtho(4, -4, 6, 0, 0, 0, 0, 0, 1)
        #glOrtho(-5, 5, -5, 5, -10, 10)
        #glOrtho(0, 1, 0, 1, -1, 1)
        # Draw X, Y, Z axis arrows
        self.draw_axis_arrow('x', 2.0, (1, 0, 0))  # Red X
        self.draw_axis_arrow('y', 2.0, (0, 1, 0))  # Green Y
        self.draw_axis_arrow('z', 2.0, (0, 0, 1))  # Blue Z

        glPushMatrix()

        # Apply rotations (order: yaw → pitch → roll)
        glTranslatef(self.shift_x, self.shift_y, 0)
        glRotatef(self.roll, 0, 1, 0)  # Yaw (around Y)
        glRotatef(self.pitch, 1, 0, 0)  # Pitch (around X)
        glRotatef(self.yaw, 0, 0, 1)  # Roll (around Z)


        # Draw shaft
        glColor4f(0.0, 1.0, 0.0, 0.5)
        shaft = gluNewQuadric()
        gluCylinder(shaft, 0.2, 0.2, 2, 20, 1)
        gluDeleteQuadric(shaft)

        # Draw head at the tip
        glRotatef(90, 0, 1, 0)
        glTranslatef(-0.0, 0, -0.75)
        glColor4f(1.0, 0.0, 0.0, 0.5)
        head = gluNewQuadric()
        gluCylinder(head, 0.3, 0.3, 1.5, 20, 1)
        gluDeleteQuadric(head)

        glPopMatrix()






class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            value = QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                int(event.position().x()),
                self.width(),
                False,
            )
            self.setValue(value)
            event.accept()
        super().mousePressEvent(event)


class UltrasoundViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.volume = volume
        # self.volume.scale('GPU', 0.1)
        # self.volume.pad('GPU')

        self.z_index = 0  # show all the slices # self.volume.z_dim // 2
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0

        self.display_slice = cp.asnumpy(0) #self.volume.rotated_gpu[:, :, self.z_index]


        #self.ui_value_reset()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.increment_slider)
        # self.timer.start(400)

        self.setWindowTitle("Ultrasound Viewer")
        self.setGeometry(100, 100, 700, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.layout = QVBoxLayout()

        # # Create a container widget
        # self.overlay_container = QWidget(self)
        # self.overlay_container.setFixedSize(256, 256)
        #
        # # Stack layout: background image and OpenGL widget
        # self.stack = QStackedLayout()
        # self.overlay_container.setLayout(self.stack)

        # self.image_label = QLabel(self)
        # self.image_label.setFixedSize(512, 512)  # Force QLabel to be exactly 512x512
        # self.image_label.setStyleSheet("border: 1px solid #666;background-color:black;")
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #
        # # Load and display image
        # pixmap = QPixmap("../data/test.jpg")
        # if not pixmap.isNull():
        #     # Optionally scale the image to fit the label
        #     pixmap = pixmap.scaled(
        #         self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        #     )
        #     self.image_label.setPixmap(pixmap)
        # else:
        #     self.image_label.setText("Image not found")


      #  self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # OpenGL Widget
        self.gl_widget = ArrowGLWidget()
        self.gl_widget.setFixedSize(512, 512)
        self.gl_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.gl_widget.setStyleSheet("background: transparent;")  # Optional
        self.layout.addWidget(self.gl_widget)

        # Add to stacked layout
        #self.stack.addWidget(self.image_label)
    #    self.stack.addWidget(self.gl_widget)

        # Add to main layout
      #  self.layout.addWidget(self.gl_widget)

        self.slider_z = self.add_slider('Slice Z', 0, 0, self.z_index, 1)
        self.slider_yaw = self.add_slider('Yaw', -180, 180, self.yaw, 1)
        self.slider_pitch = self.add_slider('Pitch', -180, 180, self.pitch, 1)
        self.slider_roll = self.add_slider('Roll', -180, 180, 0, 1)
        self.slider_x = self.add_slider('Move X', -100, 100, self.shift_x, 10)
        self.slider_y = self.add_slider('Move Y', -100, 100, self.shift_y, 10)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.ui_value_reset)
        self.layout.addWidget(self.reset_button)

        central_widget.setLayout(self.layout)

        self.db_timer = QTimer(self)
        self.db_timer.timeout.connect(self.fetch_orientation_from_db)
        self.db_timer.start(100)  # in milliseconds, e.g., 100 ms = 10 Hz



     #   self.update_image()

    def fetch_orientation_from_db(self):

        result = mysql_interface.db_select(None,'SELECT roll, pitch, yaw FROM ble_receiver.probe_imu where idx = 1', [])
        if len(result) == 1:
            print(result[0])

            self.yaw = result[0]['yaw']
            self.pitch = result[0]['pitch']
            self.roll = -result[0]['roll']

            #new_value = (self.slider_roll.value() + 10) % (self.slider_roll.maximum() + 1)
            self.slider_yaw.setValue(int(result[0]['yaw']))
            self.slider_pitch.setValue(int(result[0]['pitch']))
            self.slider_roll.setValue(int(-result[0]['roll']))


            self.gl_widget.update()

    def ui_value_reset(self):

        self.z_index = 0  # show all the slices # self.volume.z_dim // 2
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0

        # this works
        self.slider_z.setValue(self.z_index)
        self.slider_yaw.setValue(0)
        self.slider_pitch.setValue(0)
        self.slider_roll.setValue(0)
        self.slider_x.setValue(0)
        self.slider_y.setValue(0)


    def add_slider(self, name, min, max, value, step):
        slider_layout = QHBoxLayout()

        # left label
        label = QLabel(name, self)
        font = QFont()
        font.setPointSize(12)
        label.setFont(font)
        label.setFixedWidth(80)  # Set the width to 100px
        slider_layout.addWidget(label)

        #actual slider
        slider = ClickableSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(value)
        slider.setTickInterval(step)
        slider.valueChanged.connect(self.update_image)
        slider_layout.addWidget(slider)

        #right label
        label_right = QLabel(str(slider.value()), self)
        label_right.setMinimumWidth(50)  # Adjust as needed
        label_right.setFont(font)
        slider_layout.addWidget(label_right)
        slider.valueChanged.connect(lambda value: label_right.setText(str(value)))

        self.layout.addLayout(slider_layout)
        return slider


    def update_image(self):
    #
    #     start_time = time.time()
    #
    #     self.z_index = self.slider_z.value()
        self.yaw = self.slider_yaw.value()
        self.pitch = self.slider_pitch.value()
        self.roll = self.slider_roll.value()
        self.shift_x = self.slider_x.value()
        self.shift_y = self.slider_y.value()
    #
    #    # self.volume.rotate(self.yaw, self.pitch, self.roll, self.z_index, method='GPU')
    #
    #     # elapsed_time = time.time() - start_time
    #     # print(f"GPU rotate 1 Elapsed time: {elapsed_time:.3f} seconds")
    #     #start_time = time.time()
    #
    #   #  a = self.volume.rotated_gpu[:, :, self.z_index]
    #
    #   #  self.get_display_slice()
    #    # self.display_slice = self.get_display_slice()
    #    # cp.cuda.Device(0).synchronize()
    #   #  self.display_slice = cp.ascontiguousarray(self.display_slice).copy()
    #    # self.display_slice = cp.asnumpy(self.display_slice)
    #     #self.display_slice = cp.asnumpy(self.display_slice, stream=cp.cuda.get_current_stream())
    #
    #     # elapsed_time = time.time() - start_time
    #     # print(f"GPU rotate 2 Elapsed time: {elapsed_time:.3f} seconds")
    #     # start_time = time.time()
    #
    #     #self.display_slice = cp.asnumpy(self.volume.rotated_gpu[:, :, self.z_index])
    #
    #     #self.display_slice = self.shift_image(self.display_slice, self.shift_x, self.shift_y)
    #
    #     # elapsed_time = time.time() - start_time
    #     # print(f"GPU rotate 3 Elapsed time: {elapsed_time:.3f} seconds")
    #     # start_time = time.time()
    #
    #     moved = self.display_slice.copy()
    #
    #     moved = moved.astype(cp.int32)
    #     moved = cp.clip(moved, 0, None) #remove less than 0 value, which will create problem in division and cv2 function
    #     max_val = cp.max(moved)
    #
    #     if max_val == 0: #to prevent divide by 0
    #         max_val = 1
    #     img_rgb = cv2.cvtColor((moved * 255 / max_val).astype(cp.uint8), cv2.COLOR_GRAY2RGB)
    #
    #     #double the size
    #     height, width = img_rgb.shape[:2]
    #     new_size = (width * 2, height * 2)
    #     img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_LINEAR)
    #
    #
    #     qt_pixmap = self.numpy_to_qpixmap(cv2.flip(img_rgb, 0))
    #    # self.image_label.setPixmap(qt_pixmap)
    #
    #     # self.gl_widget.initializeGL()
    #     self.gl_widget.update_texture(self.volume.rotated_gpu, self.z_index)
    #     # # Redraw the scene
    #     self.gl_widget.update()
    #
    #     # elapsed_time = time.time() - start_time
    #     # print(f"GPU rotate 4 Elapsed time: {elapsed_time:.3f} seconds")
        self.gl_widget.pitch = self.pitch

        self.gl_widget.yaw = self.yaw
        self.gl_widget.roll = self.roll

        self.gl_widget.shift_x = self.slider_x.value()
        self.gl_widget.shift_y = self.slider_y.value()

        self.gl_widget.update()

    def increment_slider(self):
        new_value = (self.slider_roll.value() + 10) % (self.slider_roll.maximum() + 1)
        self.slider_roll.setValue(new_value)


    def numpy_to_qpixmap(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        return QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888))



if __name__ == "__main__":
    app = QApplication(sys.argv)
 #   volume = Volume("../data/patient0006_4CH_half_sequence.nii")
    viewer = UltrasoundViewer()
    viewer.show()
    sys.exit(app.exec())