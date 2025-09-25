import sys
import nibabel as nib
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QStyle, QHBoxLayout,\
    QSpacerItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QFont
from scipy.ndimage import rotate

import cupy as cp
import cupyx.scipy.ndimage as cndimage
import scipy.ndimage as ndimage
from scipy.ndimage import zoom

import time

class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        """Make slider jump to the clicked position."""
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

        # Load the NIfTI file
        nii_img = nib.load("D:/Ultrasound Data/CAMUS_public/patient0006_4CH_half_sequence.nii")  # Replace with your file path
        self.volume = nii_img.get_fdata()  # Convert to NumPy array
        self.volume_updated = self.volume.copy()

        # use GPU to resize and pad start-------------
        self.volume_gpu = cp.array(self.volume)


        self.volume_scale('GPU', 0.1)

        self.volume_pad('GPU')

        self.rotated_gpu = self.volume_gpu.copy()

        self.x_dim, self.y_dim, self.z_dim = self.volume_updated.shape

        # Default transformation values
        self.z_index = self.z_dim // 2
       # self.z_index = 1
        self.yaw = 0
        self.pitch = 0
        self.roll = 5
        self.shift_x = 0
        self.shift_y = 0

        self.display_slice = cp.asnumpy(self.rotated_gpu[:, :, self.z_index])

        # Initialize UI
        self.init_ui()

    def volume_scale(self, method='GPU', scale_factor=0.2):

        # order=1 for bilinear interpolation
        # order=3 for cubic interpolation
        if method == 'GPU':
            self.volume_gpu = cndimage.zoom(self.volume_gpu, zoom=scale_factor, order=0)
            print(self.volume_gpu.shape)
           # self.gpu_to_cpu()
        else:
            self.volume_updated = zoom(self.volume, zoom=(scale_factor, scale_factor, scale_factor), order=3)
            print(self.volume_updated.shape)

    def volume_pad(self, method='GPU'):

        # Find the maximum size among all dimensions
        max_dim = max(self.volume_gpu.shape) * 1.5

        # Calculate padding for each axis to match max_dim
        pad_width = [((max_dim - s) // 2, (max_dim - s + 1) // 2) for s in self.volume_gpu.shape]
        # pad_width = [(s // 2, s // 2) for s in self.volume.shape]

        if method == 'GPU':

            # Convert to CuPy tuple format
            pad_width_gpu = tuple((cp.int32(p[0]), cp.int32(p[1])) for p in pad_width)

            print(self.volume_gpu.shape)

            # Apply padding on GPU
            self.volume_gpu = cp.pad(self.volume_gpu, pad_width_gpu, mode='constant', constant_values=0)
            a = self.volume_gpu[:,:,0]
            a = self.volume_gpu[:, :, 1]

            print(self.volume_gpu.shape)

            # Move the padded volume back to CPU
            self.gpu_to_cpu()
            #self.volume_updated = cp.asnumpy(self.volume_gpu)
        else:
            # Apply padding with zeros
            self.volume_updated = np.pad(self.volume_updated, pad_width, mode='constant', constant_values=0)

    def gpu_to_cpu(self):

        # # Move the padded volume back to CPU
        # self.volume_updated = cp.asnumpy(self.volume_gpu)

        start_time = time.time()
        self.volume_updated = cp.asnumpy(self.volume_gpu)
        end_time = time.time()

        # Calculate elapsed time in seconds
        elapsed_time = end_time - start_time
        print(f"GPU gpu_to_cpu Elapsed time: {elapsed_time:.3f} seconds")


    def volume_rotate(self, angle, axes, method='GPU'):
        #img_gpu = cp.array(image)  # Move data to GPU
        self.rotated_gpu = cndimage.rotate(self.volume_gpu, angle, axes=axes, reshape=False, mode='nearest')
        print(f"GPU rotate")

        #self.display_image = self.rotated_gpu[:, :, self.z_index]

        start_time = time.time()
        #self.volume_updated = cp.asnumpy(self.rotated_gpu)
        self.display_slice = cp.asnumpy(self.rotated_gpu[:, :, self.z_index])

        end_time = time.time()

        # Calculate elapsed time in seconds
        elapsed_time = end_time - start_time
        print(f"GPU gpu_to_cpu Elapsed time: {elapsed_time:.3f} seconds")
        #self.volume_updated = cp.asnumpy(self.rotated_gpu)

    def add_slider(self, name, min, max, value, step):

        slider_layout = QHBoxLayout()

        # Left label: "Slice"
        label_left = QLabel(name, self)
        font = QFont()
        font.setPointSize(12)  # Set font size (adjust as needed)
        label_left.setFont(font)

        label_left.setMinimumWidth(30)  # Adjust as needed

        slider_layout.addWidget(label_left)

        slider_layout.addSpacerItem(QSpacerItem(20, 0))  # 20px horizontal space

        # Slice slider
        #elf.slider_z = self.create_slider(0, self.z_dim - 1, self.z_index, "Slice")
        slider = self.create_slider(min, max, value, name, step)
        slider_layout.addWidget(slider)

        label_right = QLabel(str(slider.value()), self)
        label_right.setMinimumWidth(50)  # Adjust as needed
        label_right.setFont(font)
        slider_layout.addWidget(label_right)

        self.layout.addLayout(slider_layout)

        slider.valueChanged.connect(lambda value: label_right.setText(str(value)))

        return slider


    def init_ui(self):
        self.setWindowTitle("Ultrasound Viewer")
        self.setGeometry(100, 100, 600, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        #######################
        # Slider layout (Labels + Slider)

        self.slider_z = self.add_slider('Slice', 0, self.z_dim - 1, self.z_index, 1)

        # Rotation sliders
        self.slider_yaw = self.add_slider('Yaw', -90, 90, self.yaw, 1)

        self.slider_pitch = self.add_slider('Pitch', -90, 90, self.pitch, 1)

        self.slider_roll = self.add_slider('Roll', -180, 180, self.roll, 1)

        # Translation sliders
        # self.slider_x = self.create_slider(-100, 100, self.shift_x, "Move X")
        # self.layout.addWidget(self.slider_x)
        self.slider_x = self.add_slider('Move X', -100, 100, self.shift_x, 10)


        # self.slider_y = self.create_slider(-100, 100, self.shift_y, "Move Y")
        # self.layout.addWidget(self.slider_y)
        self.slider_y = self.add_slider('Move Y', -100, 100, self.shift_y, 10)

        # Set layout
        central_widget.setLayout(self.layout)

        # Initial image display
        self.update_image()

    def create_slider(self, min_val, max_val, init_val, label_text, step=1):
        """Helper function to create a slider"""
      #  slider = QSlider(Qt.Orientation.Horizontal)

        slider = ClickableSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setTickInterval(step)
        #slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        #slider.setSingleStep(1)  # Step when using keyboard arrows
        #slider.setPageStep(1)  # Step when clicking on the track
        slider.valueChanged.connect(self.update_image)
        return slider

    def update_image(self):
        """Update the displayed image based on current slider values"""
        # Get slider values
        self.z_index = self.slider_z.value()
        self.yaw = self.slider_yaw.value()
        self.pitch = self.slider_pitch.value()
        self.roll = self.slider_roll.value()
        self.shift_x = self.slider_x.value()
        self.shift_y = self.slider_y.value()

        # Extract and rotate slice
        # slice_2d = self.volume[:, :, self.z_index]
        # rotated = rotate(slice_2d, self.yaw, axes=(0, 1), reshape=False, mode='nearest')
        #rotated = rotate(rotated, self.pitch, axes=(1, 0), reshape=False, mode='nearest')
        #rotated = rotate(rotated, self.roll, axes=(0, 1), reshape=False, mode='nearest')
        #rotated = slice_2d[:,:]


        # ################ GPU rotation
        #rotated = gpu_rotate(temp_volume, self.yaw, axes=(0, 1))  # Pitch (X-axis)
        # rotated = gpu_rotate(rotated, self.pitch, axes=(1, 2))  # Pitch (X-axis)
        #rotated = gpu_rotate(temp_volume, self.roll, axes=(1, 2))  # Pitch (X-axis)
        #self.volume_rotate(self.yaw, axes=(0, 1), method='GPU')  # Pitch (X-axis)
        self.volume_rotate(self.roll, axes=(1, 2), method='GPU')  # Pitch (X-axis)
        # ################

       # display_image = self.volume_updated[:, :, self.z_index]

       # display_image[display_image < 10 ] = 255  # Replace 255 with 0

        # Apply translation
        #moved = self.shift_image(rotated, self.shift_x, self.shift_y)

        moved = self.display_slice
        #moved = self.volume_updated[:, :, self.z_index]



        moved = moved.astype(int)
        moved[moved < 0] = 0
        b = np.max(moved)
        max_val = np.max(moved)

        if max_val == 0:#to prevent divide by 0
            max_val = 1
        img_rgb = cv2.cvtColor((moved * 255 / max_val).astype(np.uint8), cv2.COLOR_GRAY2RGB)



        height, width, _ = img_rgb.shape
        bytes_per_line = 3 * width
        qt_image = cv2.flip(img_rgb, 0)  # Flip vertically to match Qt coordinates
        qt_pixmap = self.numpy_to_qpixmap(qt_image)

        # Update image display
        self.image_label.setPixmap(qt_pixmap)

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

    def numpy_to_qpixmap(self, img):
        """Convert a NumPy array image (OpenCV format) to QPixmap"""
        height, width, channel = img.shape
        bytes_per_line = channel * width
        return QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888))


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UltrasoundViewer()
    window.show()
    sys.exit(app.exec())
