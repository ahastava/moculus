#! pip install cupy-cuda12x # For CUDA 12.x
#! pip install opencv-python
#! pip install nibabel
import sys
import numpy as np
import cupy as cp
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QStyle, QHBoxLayout, \
    QSpacerItem, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
import time

from volume import Volume  # Import the Volume class

# from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # Works in PyQt6
# import OpenGL.GL as gl # install PyOpenGL

# #
# class GLWidget(QOpenGLWidget):
#     def __init__(self, volume_gpu, parent=None):
#         super().__init__(parent)
#         self.texture_id = None
#         self.volume_gpu = volume_gpu  # Store 3D volume data
#
#     def initializeGL(self):
#         """ Initialize OpenGL resources """
#         self.makeCurrent()
#         gl.glEnable(gl.GL_TEXTURE_3D)
#
#         # Generate 3D texture
#         self.texture_id = gl.glGenTextures(1)
#         self.update_texture(self.volume_gpu)
#
#     def update_texture(self, data_gpu):
#         """ Uploads 3D volume data to OpenGL texture """
#         self.makeCurrent()
#         gl.glBindTexture(gl.GL_TEXTURE_3D, self.texture_id)
#         gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
#         gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
#
#         # Ensure data is contiguous and in the correct format
#         data_gpu = cp.ascontiguousarray(data_gpu).astype(cp.float32)
#         data_cpu = cp.asnumpy(data_gpu)  # Convert CuPy array to NumPy
#
#         gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F,
#                         data_cpu.shape[2], data_cpu.shape[1], data_cpu.shape[0], 0,
#                         gl.GL_RED, gl.GL_FLOAT, data_cpu)
#
#         gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
#
#     def paintGL(self):
#         """ Render a slice of the volume (Example: showing Z slice 50) """
#         gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
#         gl.glLoadIdentity()
#
#         z_slice = 50 / self.volume_gpu.shape[0]  # Normalize Z slice
#         if self.texture_id is not None:
#             gl.glBindTexture(gl.GL_TEXTURE_3D, self.texture_id)
#             gl.glBegin(gl.GL_QUADS)
#             gl.glTexCoord3f(0, 0, z_slice); gl.glVertex2f(-1, -1)
#             gl.glTexCoord3f(1, 0, z_slice); gl.glVertex2f(1, -1)
#             gl.glTexCoord3f(1, 1, z_slice); gl.glVertex2f(1, 1)
#             gl.glTexCoord3f(0, 1, z_slice); gl.glVertex2f(-1, 1)
#             gl.glEnd()
#             gl.glBindTexture(gl.GL_TEXTURE_3D, 0)


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
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        self.volume.scale('GPU', 0.1)
        self.volume.pad('GPU')

        self.z_index = 0  # show all the slices # self.volume.z_dim // 2
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0

        self.display_slice = cp.asnumpy(self.volume.rotated_gpu[:, :, self.z_index])


        #self.ui_value_reset()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.increment_slider)
        # self.timer.start(400)

        self.setWindowTitle("Ultrasound Viewer")
        self.setGeometry(100, 100, 700, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.layout = QVBoxLayout()

        # self.gl_widget = GLWidget(self.volume.rotated_gpu, self)
        # self.gl_widget.setFixedSize(512, 512)
        # #self.setCentralWidget(self.gl_widget)
        # self.layout.addWidget(self.gl_widget)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(712, 712)  # Force QLabel to be exactly 512x512
        self.image_label.setStyleSheet("border: 1px solid #666;background-color:black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.slider_z = self.add_slider('Slice Z', 0, self.volume.z_dim - 1, self.z_index, 1)
        self.slider_yaw = self.add_slider('Yaw', -180, 180, self.yaw, 1)
        self.slider_pitch = self.add_slider('Pitch', -180, 180, self.pitch, 1)
        self.slider_roll = self.add_slider('Roll', -180, 180, 0, 1)
        self.slider_x = self.add_slider('Move X', -100, 100, self.shift_x, 10)
        self.slider_y = self.add_slider('Move Y', -100, 100, self.shift_y, 10)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.ui_value_reset)
        self.layout.addWidget(self.reset_button)

        central_widget.setLayout(self.layout)
        self.update_image()

    def ui_value_reset(self):

        self.z_index = 0  # show all the slices # self.volume.z_dim // 2
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0

        # I don't know why this does not work individually
        # self.slider_z.setValue(self.z_index)
        # self.slider_yaw.setValue( self.yaw)
        # self.slider_pitch.setValue(self.pitch)
        # self.slider_roll.setValue(self.roll)
        # self.slider_x.setValue(self.shift_x)
        # self.slider_y.setValue(self.shift_y)

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

    def shift_image(self, image, shift_x, shift_y):
        """Shift image with zero padding (instead of wrapping)"""
        shifted = np.zeros_like(image)
        x_start = max(0, shift_x)
        x_end = min(image.shape[1], image.shape[1] + shift_x)
        y_start = max(0, shift_y)
        y_end = min(image.shape[0], image.shape[0] + shift_y)

        x_start_src = max(0, -shift_x)
        x_end_src = min(image.shape[1], image.shape[1] - shift_x)
        y_start_src = max(0, -shift_y)
        y_end_src = min(image.shape[0], image.shape[0] - shift_y)

        shifted[y_start:y_end, x_start:x_end] = image[y_start_src:y_end_src, x_start_src:x_end_src]
        return shifted

    def get_display_slice(self):

        start_time = time.time()

        matrix = self.volume.rotated_gpu  # Ensure it's a CuPy array

        sliced_matrix = matrix[:, :, self.z_index:]

        max_value = cp.max(sliced_matrix, axis=2)

     #   a = cp.asnumpy(first_nonzero_values)
       # b = self.raycast()

        elapsed_time = time.time() - start_time
        print(f"GPU display Elapsed time: {elapsed_time:.3f} seconds")

        self.display_slice = cp.asnumpy(max_value)
     #   return  cp.asnumpy(max_value)

    # def get_display_slice2(self):
    # I can't use the first non-zero value along the z axis, since the first >0 vale may be 0.5, a small value
    # even set the threashold won't work, because it will suppress the entire image
    # also it will create streaks of black lines artifacts
    #     matrix = self.volume.rotated_gpu  # Ensure it's a CuPy array
    #
    #     first_nonzero_idx = cp.argmax(matrix > 10, axis=2)  # Find first nonzero indices
    #
    #     x_idx = cp.arange(matrix.shape[0])[:, None]  # X indices
    #     y_idx = cp.arange(matrix.shape[1])  # Y indices
    #
    #     first_nonzero_values = matrix[x_idx, y_idx, first_nonzero_idx]
    #
    #     return first_nonzero_values


    def update_image(self):

        start_time = time.time()

        self.z_index = self.slider_z.value()
        self.yaw = self.slider_yaw.value()
        self.pitch = self.slider_pitch.value()
        self.roll = self.slider_roll.value()
        self.shift_x = self.slider_x.value()
        self.shift_y = self.slider_y.value()

        self.volume.rotate(self.yaw, self.pitch, self.roll, self.z_index, method='GPU')

        elapsed_time = time.time() - start_time
        print(f"GPU rotate 1 Elapsed time: {elapsed_time:.3f} seconds")
        start_time = time.time()

      #  a = self.volume.rotated_gpu[:, :, self.z_index]

        self.get_display_slice()
       # self.display_slice = self.get_display_slice()
       # cp.cuda.Device(0).synchronize()
      #  self.display_slice = cp.ascontiguousarray(self.display_slice).copy()
       # self.display_slice = cp.asnumpy(self.display_slice)
        #self.display_slice = cp.asnumpy(self.display_slice, stream=cp.cuda.get_current_stream())

        elapsed_time = time.time() - start_time
        print(f"GPU rotate 2 Elapsed time: {elapsed_time:.3f} seconds")
        start_time = time.time()

        #self.display_slice = cp.asnumpy(self.volume.rotated_gpu[:, :, self.z_index])

        self.display_slice = self.shift_image(self.display_slice, self.shift_x, self.shift_y)

        elapsed_time = time.time() - start_time
        print(f"GPU rotate 3 Elapsed time: {elapsed_time:.3f} seconds")
        start_time = time.time()

        moved = self.display_slice.copy()

        moved = moved.astype(cp.int32)
        moved = cp.clip(moved, 0, None) #remove less than 0 value, which will create problem in division and cv2 function
        max_val = cp.max(moved)

        if max_val == 0: #to prevent divide by 0
            max_val = 1
        img_rgb = cv2.cvtColor((moved * 255 / max_val).astype(cp.uint8), cv2.COLOR_GRAY2RGB)

        #double the size
        height, width = img_rgb.shape[:2]
        new_size = (width * 2, height * 2)
        img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_LINEAR)


        qt_pixmap = self.numpy_to_qpixmap(cv2.flip(img_rgb, 0))
        self.image_label.setPixmap(qt_pixmap)

        # self.gl_widget.initializeGL()
        #self.gl_widget.update_texture(self.volume.rotated_gpu)
        # # Redraw the scene
        #self.gl_widget.update()

        elapsed_time = time.time() - start_time
        print(f"GPU rotate 4 Elapsed time: {elapsed_time:.3f} seconds")

    def increment_slider(self):
        new_value = (self.slider_roll.value() + 10) % (self.slider_roll.maximum() + 1)
        self.slider_roll.setValue(new_value)


    def numpy_to_qpixmap(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        return QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    #D:/Ultrasound Data/CAMUS_public/patient0006_4CH_half_sequence.nii
    volume = Volume("../data/patient0006_4CH_half_sequence.nii")
    viewer = UltrasoundViewer(volume)
    viewer.show()
    sys.exit(app.exec())