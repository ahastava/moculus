

import sys
import numpy as np
import cv2
import mysql_interface # put it earlier, don't know why, but must do this

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QStyle, QHBoxLayout, \
    QPushButton, QCheckBox, QSizePolicy, QLineEdit, QFileDialog, QStyleOptionSlider, QStyle
from PyQt6.QtCore import Qt, QTimer, QCoreApplication, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QCloseEvent

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from stl import mesh
import asyncio
import threading

# --- DATABASE IMPORT ---
# This import is required for the database logic from main_opengl - hammer.py
# If mysql_interface is not available, this script will crash.


# --- BLE HANDLER IMPORT ---
import bleak_probe_receiver_handler_db_async
from bleak import BleakClient
from datetime import datetime
import logging


# --- BLE WORKER THREAD ---
class BLEWorkerThread(QThread):
    """Thread to run the async BLE connection without blocking the GUI"""

    # Signals to communicate with main thread
    imu_data_signal = pyqtSignal(int, int, int)  # roll, pitch, yaw
    ble_status_signal = pyqtSignal(str, str)  # status_message, color_code
    connection_state_signal = pyqtSignal(bool)  # True=connected, False=disconnected

    def __init__(self):
        super().__init__()
        self.running = False
        self.should_connect = False
        self.loop = None
        self.db_manager = None
        self.client = None
        self.is_connected = False

    def run(self):
        """Run the asyncio event loop in this thread"""
        self.running = True
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Run the async main function
            self.loop.run_until_complete(self.async_main())
        except Exception as e:
            logging.error(f"BLE Worker error: {e}")
            self.ble_status_signal.emit(f"Error: {str(e)}", "red")
            self.connection_state_signal.emit(False)
        finally:
            if self.loop:
                self.loop.close()

    def request_connect(self):
        """Request connection to BLE device"""
        self.should_connect = True

    def request_disconnect(self):
        """Request disconnection from BLE device"""
        self.should_connect = False

    def notification_handler(self, sender, data):
        """Handler for BLE notifications - modified to emit Qt signals"""
        try:
            text = data.decode('utf-8')
            parts = text.split(',')

            counter = int(parts[0])
            yaw = float(parts[1])
            pitch = float(parts[2])
            roll = float(parts[3])
            timestamp = datetime.now()

            print(f"BLE Data: counter {counter}, yaw {yaw:.2f}, pitch {pitch:.2f}, roll {roll:.2f}")

            # Emit signal to update GUI (thread-safe)
            self.imu_data_signal.emit(int(roll), int(pitch), int(yaw))

            if self.db_manager:
                # Fast async update of current record
                asyncio.create_task(
                    self.db_manager.update_current(counter, roll, pitch, yaw, timestamp)
                )

                # Queue for background history insert
                self.db_manager.add_to_history_queue(counter, roll, pitch, yaw, timestamp)

        except Exception as e:
            logging.error(f"Notification error: {e}")
            self.ble_status_signal.emit(f"Notification error: {str(e)}", "orange")

    async def async_main(self):
        """Async main function to connect to BLE device"""
        logging.basicConfig(level=logging.INFO)

        # Initialize database manager
        self.db_manager = bleak_probe_receiver_handler_db_async.UltraFastDatabaseManager(
            bleak_probe_receiver_handler_db_async.DB_CONFIG
        )
        await self.db_manager.create_pool()

        self.ble_status_signal.emit("Ready to connect", "orange")
        self.connection_state_signal.emit(False)

        try:
            # Main loop - handle connection/disconnection requests
            while self.running:
                if self.should_connect and not self.is_connected:
                    await self.connect_to_device()
                elif not self.should_connect and self.is_connected:
                    await self.disconnect_from_device()

                await asyncio.sleep(0.1)

        except Exception as e:
            self.ble_status_signal.emit(f"Connection error: {str(e)}", "red")
            self.connection_state_signal.emit(False)
            logging.error(f"BLE connection error: {e}")
        finally:
            if self.is_connected:
                await self.disconnect_from_device()
            if self.db_manager:
                await self.db_manager.close_pool()

    async def connect_to_device(self):
        """Connect to the BLE device"""
        try:
            self.ble_status_signal.emit("Connecting...", "orange")

            self.client = BleakClient(bleak_probe_receiver_handler_db_async.DEVICE_ADDRESS)
            await self.client.connect()

            if not self.client.is_connected:
                self.ble_status_signal.emit("Failed to connect", "red")
                self.connection_state_signal.emit(False)
                self.is_connected = False
                return

            self.is_connected = True
            self.ble_status_signal.emit("Connected", "green")
            self.connection_state_signal.emit(True)
            print("Connected to BLE device.")

            # Start notifications
            await self.client.start_notify(
                bleak_probe_receiver_handler_db_async.CHARACTERISTIC_UUID_1,
                self.notification_handler
            )
            print("Started notifications.")

        except Exception as e:
            self.ble_status_signal.emit(f"Connection failed: {str(e)}", "red")
            self.connection_state_signal.emit(False)
            self.is_connected = False
            logging.error(f"Failed to connect: {e}")

    async def disconnect_from_device(self):
        """Disconnect from the BLE device"""
        try:
            if self.client and self.client.is_connected:
                self.ble_status_signal.emit("Disconnecting...", "orange")

                # Stop notifications
                await self.client.stop_notify(
                    bleak_probe_receiver_handler_db_async.CHARACTERISTIC_UUID_1
                )

                # Disconnect
                await self.client.disconnect()
                print("BLE Client disconnected.")

            self.is_connected = False
            self.client = None
            self.ble_status_signal.emit("Disconnected", "red")
            self.connection_state_signal.emit(False)

        except Exception as e:
            self.ble_status_signal.emit(f"Disconnect error: {str(e)}", "orange")
            logging.error(f"Failed to disconnect: {e}")
            self.is_connected = False
            self.connection_state_signal.emit(False)

    def stop(self):
        """Stop the BLE worker thread"""
        self.running = False
        self.should_connect = False


# --- CUSTOM WIDGETS ---

class ClickableSlider(QSlider):
    """
    A custom QSlider that allows the user to jump to a position by clicking on the groove.
    """

    def mousePressEvent(self, event):
        """Overrides the standard mouse press event to set the value based on the click position."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate the value based on the click position
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove,
                                               self)

            if self.orientation() == Qt.Orientation.Horizontal:
                # Calculate the ratio of the click position relative to the groove length
                ratio = (event.pos().x() - rect.x()) / rect.width()
            else:
                # For vertical slider, use Y position and invert the ratio (top is max)
                ratio = (rect.bottom() - event.pos().y()) / rect.height()

            # Ensure ratio is clamped between 0 and 1
            ratio = max(0.0, min(1.0, ratio))

            # Calculate the new value and set it
            new_value = self.minimum() + int(ratio * (self.maximum() - self.minimum()))
            self.setValue(new_value)

            # Since we handled the press, we don't call the base class implementation for a standard press.
            event.accept()

        # Still allow the base class to handle other events (like dragging the handle)
        super().mousePressEvent(event)

    # --- END CUSTOM WIDGETS ---


class ArrowGLWidget(QOpenGLWidget):
    """
    OpenGL widget for rendering the 3D probe model and coordinate axes,
    and the aspect-ratio corrected, shiftable background image,
    including the textured slicing plane.
    """

    # Constants for 3D scaling and screen-aligned movement calculation
    WORLD_SHIFT_SCALE = 0.04

    # Basis vectors for screen-aligned movement, calculated for the fixed camera position (5, -5, 5)
    # R (Right/Horizontal Screen X) vector in World Coordinates
    R_X = 0.7071
    R_Y = 0.7071
    # U (Up/Vertical Screen Y) vector in World Coordinates
    U_X = -0.4082
    U_Y = 0.4082
    U_Z = 0.8165

    def initializeGL(self):
        # Configure initial OpenGL state
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Initialize transformation variables
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.shift_x = 0
        self.shift_y = 0
        self.z_index = 0
        self.is_slicing_enabled = False

        # Texture variables
        self.texture_width = 1.0
        self.texture_height = 1.0
        self.us_texture_id = 0  # New texture ID for the US clip

        # Load initial STL mesh and background texture
        try:
            # NOTE: Assuming ../data/probe_shell_500_faces.stl exists
            self.probe_mesh = mesh.Mesh.from_file('../data/probe_shell_500_faces.stl')
            # Pre-processing steps for model alignment (omitted for brevity, assume correct)
            all_verts = self.probe_mesh.vectors.reshape(-1, 3)
            min_z = np.min(all_verts[:, 2])
            self.probe_mesh.vectors[:, :, 2] -= min_z
            self.probe_mesh.vectors[:, :, 2] *= -1
            all_verts = self.probe_mesh.vectors.reshape(-1, 3)
            min_z = np.min(all_verts[:, 2])
            self.probe_mesh.vectors[:, :, 2] -= min_z
            all_verts = self.probe_mesh.vectors.reshape(-1, 3)
            mean_x = 0.5 * (np.min(all_verts[:, 0]) + np.max(all_verts[:, 0]))
            mean_y = 0.5 * (np.min(all_verts[:, 1]) + np.max(all_verts[:, 1]))
            self.probe_mesh.vectors[:, :, 0] -= mean_x
            self.probe_mesh.vectors[:, :, 1] -= mean_y
            x = self.probe_mesh.vectors[:, :, 0].copy()
            y = self.probe_mesh.vectors[:, :, 1].copy()
            self.probe_mesh.vectors[:, :, 0] = y
            self.probe_mesh.vectors[:, :, 1] = -x

            # Load the main anatomical background texture
            # NOTE: Assuming ../data/test2.jpg exists
            self.background_texture = self.load_texture("../data/test2.jpg")
        except Exception as e:
            print(f"Error loading initial assets: {e}")
            self.probe_mesh = None
            self.background_texture = 0

    def load_texture(self, path):
        """Standard texture loader for both background and US image."""
        try:
            img = QImage(path)
            # QImage loads V=0 as the top row of pixels.
            img = img.convertToFormat(QImage.Format.Format_RGBA8888)
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))

            # Store dimensions for background aspect ratio correction only
            if path.endswith("test2.jpg"):
                self.texture_width = width
                self.texture_height = height

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            # Load the data without flipping the array
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, arr)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return texture_id
        except Exception as e:
            print(f"Error loading texture from path {path}: {e}")
            return 0

    def load_us_texture(self, path):
        """Specific loader for the US fan image."""
        # Delete old texture if it exists to free GPU memory
        if self.us_texture_id != 0:
            glDeleteTextures([self.us_texture_id])

        new_id = self.load_texture(path)

        if new_id != 0:
            self.us_texture_id = new_id
        else:
            self.us_texture_id = 0

        self.update()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Set up perspective projection
        gluPerspective(45, w / h if h != 0 else 1, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def draw_axis_arrow(self, axis, length, color):
        glPushMatrix()
        glColor3f(*color)

        if axis == 'x':
            glRotatef(90, 0, 1, 0)
        elif axis == 'y':
            glRotatef(-90, 1, 0, 0)

        # Draw shaft
        quad = gluNewQuadric()
        gluCylinder(quad, 0.05, 0.05, length, 20, 1)

        # Draw arrow head (cone)
        glTranslatef(0, 0, length)
        gluCylinder(quad, 0.1, 0.0, 0.3, 20, 1)
        gluDeleteQuadric(quad)
        glPopMatrix()

    def paintGL(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --- 1. Draw Background Texture ("Contain" Mode) ---
        if self.background_texture:
            glDisable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()

            w_vp, h_vp = self.width(), self.height()
            vp_ratio = w_vp / h_vp if h_vp > 0 else 1.0
            tex_ratio = self.texture_width / self.texture_height if self.texture_height > 0 else 1.0

            # Map coordinates 0 to 1 for the viewport
            glOrtho(0, 1, 0, 1, -1, 1)

            x_start, y_start, quad_w, quad_h = 0.0, 0.0, 1.0, 1.0

            if vp_ratio > tex_ratio:
                # Viewport is wider than image: Image height fits, width is letterboxed.
                quad_w = tex_ratio / vp_ratio
                x_start = (1.0 - quad_w) / 2.0
            else:
                # Viewport is taller than image: Image width fits, height is letterboxed.
                quad_h = vp_ratio / tex_ratio
                y_start = (1.0 - quad_h) / 2.0

            quad_verts = [
                (x_start, y_start), (x_start + quad_w, y_start),
                (x_start + quad_w, y_start + quad_h), (x_start, y_start + quad_h)
            ]

            # Corrected texture coordinates for background:
            # Flips the image data (V=0 is top) to display correctly on the OpenGL quad (V=1 is top).
            tex_coords = [(0, 1), (1, 1), (1, 0), (0, 0)]

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.background_texture)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            glBegin(GL_QUADS)
            glTexCoord2f(*tex_coords[0]);
            glVertex2f(*quad_verts[0])
            glTexCoord2f(*tex_coords[1]);
            glVertex2f(*quad_verts[1])
            glTexCoord2f(*tex_coords[2]);
            glVertex2f(*quad_verts[2])
            glTexCoord2f(*tex_coords[3]);
            glVertex2f(*quad_verts[3])
            glEnd()

            glDisable(GL_TEXTURE_2D)

            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glEnable(GL_DEPTH_TEST)

        # --- 2. Camera Setup ---
        glLoadIdentity()
        # Fixed isometric camera view (5, -5, 5)
        gluLookAt(5, -5, 5, 0, 0, 0, 0, 0, 1)

        # --- 3. Draw X, Y, Z axis arrows ---
        self.draw_axis_arrow('x', 2.0, (1, 0, 0))  # Red X
        self.draw_axis_arrow('y', 2.0, (0, 1, 0))  # Green Y
        self.draw_axis_arrow('z', 2.0, (0, 0, 1))  # Blue Z

        # Calculate probe translation based on screen-aligned movement
        Sx = self.shift_x * self.WORLD_SHIFT_SCALE
        Sy = self.shift_y * self.WORLD_SHIFT_SCALE
        Z_world = self.z_index * self.WORLD_SHIFT_SCALE

        # Calculate World translation components
        x_trans = (Sx * self.R_X) + (Sy * self.U_X)
        y_trans = (Sx * self.R_Y) + (Sy * self.U_Y)
        z_trans = (Sy * self.U_Z) + Z_world

        # --- 4. Draw Slicing Plane (Textured or solid) ---
        if self.is_slicing_enabled:
            glPushMatrix()

            # Apply screen-aligned translation
            glTranslatef(x_trans, y_trans, z_trans)

            # Apply probe's rotations (Order ZXY: Yaw -> Pitch -> Roll)
            glRotatef(self.yaw, 0, 0, 1)
            glRotatef(self.pitch, 1, 0, 0)
            glRotatef(self.roll, 0, 1, 0)

            # Constants for the fan geometry
            FAN_LENGTH = 4.0
            FAN_BASE_WIDTH = 1.0
            FAN_MAX_WIDTH = 5.0

            # --- Texturing/Color Setup ---
            if self.us_texture_id != 0:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.us_texture_id)
                # Use near-white color for slight transparency
                glColor4f(1.0, 1.0, 1.0, 0.95)
            else:
                glDisable(GL_TEXTURE_2D)
                # Fallback to solid semi-transparent green
                glColor4f(0.0, 0.8, 0.0, 0.5)

            glDisable(GL_CULL_FACE)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)

            # Calculate near-end texture coverage (maps the center of the texture to the base)
            texture_coverage_factor = FAN_BASE_WIDTH / FAN_MAX_WIDTH
            u_near_left = 0.5 - (0.5 * texture_coverage_factor)
            u_near_right = 0.5 + (0.5 * texture_coverage_factor)

            glBegin(GL_QUADS)

            # CORRECTED US Slicing Plane Texture Coords:
            # V=0.0 (Top of texture/Near Field) must map to Z=0 (Probe face).
            # V=1.0 (Bottom of texture/Far Field) must map to Z=-FAN_LENGTH (Far plane).

            # V1: Near, Left, Z=0 (Probe Face)
            glTexCoord2f(u_near_left, 0.0);
            glVertex3f(-FAN_BASE_WIDTH / 2, 0, 0)

            # V2: Near, Right, Z=0 (Probe Face)
            glTexCoord2f(u_near_right, 0.0);
            glVertex3f(FAN_BASE_WIDTH / 2, 0, 0)

            # V3: Far, Right, Z=-FAN_LENGTH (Far Plane)
            glTexCoord2f(1.0, 1.0);
            glVertex3f(FAN_MAX_WIDTH / 2, 0, -FAN_LENGTH)

            # V4: Far, Left, Z=-FAN_LENGTH (Far Plane)
            glTexCoord2f(0.0, 1.0);
            glVertex3f(-FAN_MAX_WIDTH / 2, 0, -FAN_LENGTH)

            glEnd()

            if self.us_texture_id != 0:
                glDisable(GL_TEXTURE_2D)

            glDisable(GL_POLYGON_OFFSET_FILL)
            glEnable(GL_CULL_FACE)
            glPopMatrix()

            # --- 5. Draw Probe Mesh (Shifted and rotated) ---
        if self.probe_mesh is not None:
            glPushMatrix()

            # Apply screen-aligned translation
            glTranslatef(x_trans, y_trans, z_trans)

            # Apply rotations (Order: Yaw → Pitch → Roll)
            glRotatef(self.yaw, 0, 0, 1)
            glRotatef(self.pitch, 1, 0, 0)
            glRotatef(self.roll, 0, 1, 0)

            glScalef(0.01, 0.01, 0.01)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(1.0, 0.5, 0.2, 0.7)  # Semi-transparent orange

            # Draw STL triangles
            glBegin(GL_TRIANGLES)
            for i in range(len(self.probe_mesh.vectors)):
                normal = self.probe_mesh.normals[i]
                glNormal3f(*normal)
                for vertex in self.probe_mesh.vectors[i]:
                    glVertex3f(*vertex)
            glEnd()

            glPopMatrix()


class UltrasoundViewer(QMainWindow):
    """Main application window for the Ultrasound Viewer."""

    # Define PyQt signals for thread-safe communication from BLE handler
    imu_data_signal = pyqtSignal(int, int, int)  # roll, pitch, yaw
    ble_status_signal = pyqtSignal(str, str)  # status_message, color_code

    def __init__(self):
        super().__init__()

        # --- Window Setup for Scalability ---
        self.setWindowTitle("Probe Orientation & Shift Viewer")
        # Set a default, non-fullscreen size
        self.setGeometry(100, 100, 1200, 800)
        # self.showFullScreen() # REMOVED for standard scalable window

        # Initial state setup
        self.z_index = 0
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0
        self.is_slicing_enabled = False
        # New state variable: True means DB updates sliders (Live Mode); False means sliders are Manual
        self.is_live_data_mode = False

        # --- BLE Integration Setup ---
        # Create BLE worker thread
        # self.ble_worker = BLEWorkerThread()
        # self.ble_worker.imu_data_signal.connect(self.handle_imu_data)
        # self.ble_worker.ble_status_signal.connect(self.handle_ble_status)
        # self.ble_worker.connection_state_signal.connect(self.handle_connection_state)
        #
        # # # Start BLE worker thread (but don't auto-connect)
        # self.ble_worker.start()

        # Add a status label for BLE connection feedback
        self.ble_status_label = None  # Will be created in UI setup
        self.probe_connect_button = None
        self.probe_disconnect_button = None
        self.connection_indicator = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # --- Professional UI Styling (Global) ---
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #eef4f9;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                min-height: 28px;
                box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLabel {
                font-size: 13px;
            }
            .QGroupBox {
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                margin-top: 8px;
                background-color: #ffffff;
            }
            .QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: transparent;
                color: #3498db;
                font-weight: bold;
                font-size: 14px;
            }
            QSlider::handle:horizontal {
                background: #e74c3c;
                border: 1px solid #c0392b;
                width: 16px;
                height: 16px;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #bdc3c7;
                margin: 3px 0;
                border-radius: 3px;
            }
        """)

        # 1. Main layout is HORIZONTAL: GL Widget on left, Controls on right
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # 1. GL Widget (Left Section - Dominant size)
        self.gl_widget = ArrowGLWidget()
        # Set size policy to expand horizontally and vertically
        self.gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gl_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        main_layout.addWidget(self.gl_widget, stretch=4)  # Stretch factor ensures it takes most space

        # 2. Control Panel (Right Section - Fixed width for consistent layout)
        control_panel = QWidget()
        # Set a fixed minimum width, but allow it to shrink if absolutely necessary
        control_panel.setMinimumWidth(350)
        control_panel.setMaximumWidth(450)
        control_panel.setSizePolicy(QSizePolicy.Policy.Fixed,
                                    QSizePolicy.Policy.Expanding)  # Keep width fixed relative to main app size

        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(12, 12, 12, 12)

        header_label = QLabel("3D Transformation Controls")
        header_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #3498db; margin-bottom: 3px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(header_label)

        # --- PROBE CONNECTION CONTROL GROUP ---
        connection_group = QWidget()
        connection_group.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 2px solid #3498db;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        connection_layout = QVBoxLayout(connection_group)
        connection_layout.setContentsMargins(8, 8, 8, 8)
        connection_layout.setSpacing(6)

        # Connection header
        connection_header = QLabel("Probe BLE Connection")
        connection_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #3498db; border: none;")
        connection_layout.addWidget(connection_header)

        # Status indicator and label container
        status_container = QHBoxLayout()
        status_container.setSpacing(8)

        # Visual connection indicator (circle)
        self.connection_indicator = QLabel("●")
        self.connection_indicator.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #e74c3c;
                border: none;
                padding: 0;
            }
        """)
        self.connection_indicator.setFixedWidth(25)
        self.connection_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_container.addWidget(self.connection_indicator)

        # BLE status label
        self.ble_status_label = QLabel("Status: Ready")
        self.ble_status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                border: none;
                padding: 3px;
            }
        """)
        status_container.addWidget(self.ble_status_label, stretch=1)

        connection_layout.addLayout(status_container)

        # Connection buttons container
        button_container = QHBoxLayout()
        button_container.setSpacing(8)

        # Connect button
        self.probe_connect_button = QPushButton("Connect Probe")
        self.probe_connect_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.probe_connect_button.clicked.connect(self.connect_probe)
        button_container.addWidget(self.probe_connect_button)

        # Disconnect button
        self.probe_disconnect_button = QPushButton("Disconnect")
        self.probe_disconnect_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.probe_disconnect_button.clicked.connect(self.disconnect_probe)
        self.probe_disconnect_button.setEnabled(False)
        button_container.addWidget(self.probe_disconnect_button)

        connection_layout.addLayout(button_container)
        control_layout.addWidget(connection_group)
        control_layout.addSpacing(6)

        # --- LIVE DATA/MANUAL CONTROL TOGGLE ---
        self.live_data_checkbox = QCheckBox("Live Data Control (BLE)")
        self.live_data_checkbox.setChecked(self.is_live_data_mode)
        self.live_data_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: 700;
                color: #27ae60; /* Green */
                padding: 6px;
                border: 2px solid #2ecc71;
                border-radius: 5px;
                background-color: #f0fdf4;
                min-height: 28px;
            }
            QCheckBox:hover {
                background-color: #eaf8f0;
            }
            /* FIX: Make the indicator box visible */
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: white;
                border: 2px solid #2ecc71; /* Green border */
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #2ecc71; /* Green background when checked */
            }
        """)
        self.live_data_checkbox.stateChanged.connect(self.toggle_live_manual_mode)
        control_layout.addWidget(self.live_data_checkbox)

        # --- Slicing Checkbox ---
        self.slicing_checkbox = QCheckBox("Enable Probe Slicing Plane")
        self.slicing_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: 700;
                color: #e74c3c;
                padding: 6px;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                background-color: #fceceb;
                min-height: 28px;
            }
            QCheckBox:hover {
                background-color: #f7dddb;
            }
            /* FIX: Make the indicator box visible */
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: white;
                border: 2px solid #e74c3c; /* Red border */
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #e74c3c; /* Red background when checked */
            }
        """)
        self.slicing_checkbox.stateChanged.connect(self.update_slicing_state)
        control_layout.addWidget(self.slicing_checkbox)

        # --- US Clip Image Loader Group ---
        image_load_group = QWidget()
        image_load_group.setObjectName("QGroupBox")
        image_load_layout = QVBoxLayout(image_load_group)
        image_load_layout.setContentsMargins(8, 12, 8, 8)

        load_header = QLabel("US Image Texture Mapping")
        load_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #3498db;")
        image_load_layout.addWidget(load_header)

        self.us_image_path_edit = QLineEdit(self)
        self.us_image_path_edit.setPlaceholderText("Enter fan-shaped image path or browse...")
        self.us_image_path_edit.setMinimumHeight(26)
        image_load_layout.addWidget(self.us_image_path_edit)

        # File selection buttons
        button_container_img = QHBoxLayout()
        self.select_file_button = QPushButton("Browse")
        self.select_file_button.clicked.connect(self.open_file_dialog)
        self.load_image_button = QPushButton("Load Texture")
        self.load_image_button.clicked.connect(self.load_us_image)
        button_container_img.addWidget(self.select_file_button)
        button_container_img.addWidget(self.load_image_button)
        image_load_layout.addLayout(button_container_img)

        control_layout.addWidget(image_load_group)
        control_layout.addSpacing(8)

        # --- Rotation Group ---
        rotation_group_widget = QWidget()
        rotation_group_widget.setObjectName("QGroupBox")
        rotation_group = QVBoxLayout(rotation_group_widget)
        rotation_group.setContentsMargins(8, 12, 8, 8)

        rotation_header = QLabel("Orientation (Rotation)")
        rotation_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #3498db;")
        rotation_group.addWidget(rotation_header)

        container_yaw, self.slider_yaw = self.add_slider('Yaw (Z-Axis)', -180, 180, self.yaw, 5)
        rotation_group.addWidget(container_yaw)
        container_pitch, self.slider_pitch = self.add_slider('Pitch (X-Axis)', -45, 45, self.pitch, 1)
        rotation_group.addWidget(container_pitch)
        container_roll, self.slider_roll = self.add_slider('Roll (Y-Axis)', -90, 90, self.roll, 1)
        rotation_group.addWidget(container_roll)

        control_layout.addWidget(rotation_group_widget)
        control_layout.addSpacing(8)

        # --- Shift/Slice Group ---
        shift_group_widget = QWidget()
        shift_group_widget.setObjectName("QGroupBox")
        shift_group = QVBoxLayout(shift_group_widget)
        shift_group.setContentsMargins(8, 12, 8, 8)

        shift_header = QLabel("Probe Position (Shift - Screen Aligned)")
        shift_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #3498db;")
        shift_group.addWidget(shift_header)

        # --- Shift/Slice Group ---
        shift_group_widget = QWidget()
        shift_group_widget.setObjectName("QGroupBox")
        shift_group = QVBoxLayout(shift_group_widget)
        shift_group.setContentsMargins(10, 20, 10, 10)

        shift_header = QLabel("Probe Position (Shift - Screen Aligned)")
        shift_header.setStyleSheet("font-size: 16px; font-weight: 600; color: #3498db;")
        shift_group.addWidget(shift_header)

        container_x, self.slider_x = self.add_slider('Shift X (Screen Horiz.)', -100, 100, self.shift_x, 10)
        shift_group.addWidget(container_x)
        container_y, self.slider_y = self.add_slider('Shift Y (Screen Vert.)', -100, 100, self.shift_y, 10)
        shift_group.addWidget(container_y)

        container_z, self.slider_z = self.add_slider('Slice Z (World Vert.)', -50, 50, self.z_index, 5)
        shift_group.addWidget(container_z)

        control_layout.addWidget(shift_group_widget)

        # --- Reset Button ---
        control_layout.addSpacing(10)
        self.reset_button = QPushButton("Reset All Transformations", self)
        self.reset_button.clicked.connect(self.ui_value_reset)
        control_layout.addWidget(self.reset_button)

        # --- Close Button ---
        self.close_button = QPushButton("Close Application", self)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; /* Red color for warning/exit */
                color: white;
                font-size: 14px;
                font-weight: 700;
                padding: 10px 16px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        self.close_button.clicked.connect(self.close)  # Connects to QMainWindow's close method
        control_layout.addWidget(self.close_button)

        control_layout.addStretch(1)  # Push all controls up

        main_layout.addWidget(control_panel, stretch=1)

        # --- DB Timer Setup (New Logic) ---
        self.db_timer = QTimer(self)
        self.db_timer.timeout.connect(self.fetch_orientation_from_db)
        self.db_timer.start(100)  # Update every 100 ms (10 Hz)

        # Initial call to set the slider enabled state
        self.toggle_live_manual_mode(Qt.CheckState.Unchecked.value)

    def handle_imu_data(self, roll, pitch, yaw):
        """
        Handle incoming IMU data from BLE receiver.
        Updates orientation values when in live data mode.

        Args:
            roll (int): Roll angle in degrees
            pitch (int): Pitch angle in degrees
            yaw (int): Yaw angle in degrees
        """
        # if not self.is_live_data_mode:
        #     return  # Ignore IMU data if not in live mode

        try:
            # Update internal state - invert roll and pitch for live mode
            self.yaw = yaw
            self.pitch = -pitch  # Inverted for live mode
            self.roll = roll  # Inverted for live mode (was negated before)

            # Block signals to prevent triggering update_image
            self.slider_yaw.blockSignals(True)
            self.slider_pitch.blockSignals(True)
            self.slider_roll.blockSignals(True)

            # Update sliders without triggering valueChanged signal
            self.slider_yaw.setValue(int(yaw))
            self.slider_pitch.setValue(int(-pitch))  # Inverted for live mode
            self.slider_roll.setValue(int(roll))  # Inverted for live mode

            # Unblock signals
            self.slider_yaw.blockSignals(False)
            self.slider_pitch.blockSignals(False)
            self.slider_roll.blockSignals(False)

            # Update GL widget directly
            self.gl_widget.yaw = self.yaw
            self.gl_widget.pitch = self.pitch
            self.gl_widget.roll = self.roll
            self.gl_widget.update()

            print(f"BLE IMU Update: Yaw={yaw}, Pitch={-pitch}, Roll={roll}")

        except (ValueError, TypeError) as e:
            print(f"Error processing IMU data: {e}")

    def handle_ble_status(self, status_message, color_code):
        """
        Handle BLE connection status updates.

        Args:
            status_message (str): Status message describing BLE connection state
            color_code (str): Color code for status indication (e.g., 'green', 'red', 'orange')
        """
        # Update status label if it exists
        if self.ble_status_label is not None:
            self.ble_status_label.setText(f"Status: {status_message}")

            # Change color based on color_code
            color_map = {
                'green': "#27ae60",
                'red': "#e74c3c",
                'orange': "#f39c12",
                'yellow': "#f1c40f"
            }

            color = color_map.get(color_code.lower(), "#2c3e50")
            self.ble_status_label.setStyleSheet(
                f"QLabel {{ color: {color}; font-weight: bold; font-size: 13px; border: none; padding: 5px; }}"
            )

        # Print status to console for debugging
        print(f"BLE Status Update: {status_message} (Color: {color_code})")

    def handle_connection_state(self, is_connected):
        """
        Handle BLE connection state changes.

        Args:
            is_connected (bool): True if connected, False if disconnected
        """
        if is_connected:
            # Update indicator to green
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: #27ae60;
                    border: none;
                    padding: 0;
                }
            """)
            # Enable disconnect button, disable connect button
            self.probe_connect_button.setEnabled(False)
            self.probe_disconnect_button.setEnabled(True)

        else:
            # Update indicator to red
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: #e74c3c;
                    border: none;
                    padding: 0;
                }
            """)
            # Enable connect button, disable disconnect button
            self.probe_connect_button.setEnabled(True)
            self.probe_disconnect_button.setEnabled(False)

            # If live mode is enabled, disable it when disconnected
            if self.is_live_data_mode:
                self.live_data_checkbox.setChecked(False)

    def connect_probe(self):
        """Request connection to the probe"""
        print("Requesting probe connection...")
        self.ble_worker.request_connect()
        self.probe_connect_button.setEnabled(False)  # Prevent multiple clicks

    def disconnect_probe(self):
        """Request disconnection from the probe"""
        print("Requesting probe disconnection...")
        self.ble_worker.request_disconnect()
        self.probe_disconnect_button.setEnabled(False)  # Prevent multiple clicks

    def toggle_live_manual_mode(self, state):
        """Toggles between reading orientation from BLE (Live) and using sliders (Manual)."""
        # Check if user is trying to enable live mode without connection
        # if state == Qt.CheckState.Checked.value : #and self.ble_worker is not None and not self.ble_worker.is_connected:
        #     # Show warning message
        #     from PyQt6.QtWidgets import QMessageBox
        #     msg = QMessageBox(self)
        #     msg.setIcon(QMessageBox.Icon.Warning)
        #     msg.setWindowTitle("Probe Not Connected")
        #     msg.setText("Cannot enable Live Data Mode")
        #     msg.setInformativeText("Please connect to the probe first using the 'Connect Probe' button above.")
        #     msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        #     msg.exec()
        #
        #     # Uncheck the checkbox
        #     self.live_data_checkbox.setChecked(False)
        #     return

        self.is_live_data_mode = (state == Qt.CheckState.Checked.value)

        # Enable/Disable the rotation sliders based on the mode.
        # If Live (True), disable sliders so BLE controls them.
        # If Manual (False), enable sliders so the user controls them.
        is_manual = not self.is_live_data_mode
        self.slider_yaw.setEnabled(is_manual)
        self.slider_pitch.setEnabled(is_manual)
        self.slider_roll.setEnabled(is_manual)

        # The shift/slice sliders are always controlled by the user, so they remain enabled.
    def fetch_orientation_from_db(self):
        """
        Fetches probe orientation (roll, pitch, yaw) from the SQL database and updates the UI.

        This only updates the rotation sliders if in LIVE mode.
        """
        # --- NEW LOGIC: Only update rotation if in Live Data Mode ---

        if not self.is_live_data_mode:
            return
            # -----------------------------------------------------------
        #print("fetch_orientation_from_db")
        # The query and structure are copied directly from main_opengl - hammer.py
        try:
            result = mysql_interface.db_select(None, 'SELECT roll, pitch, yaw FROM ble_receiver.probe_imu where idx = 1',
                                               [])

          #  result = [{'yaw':0,'pitch':0, "roll":0}]
          #  print(result)
            if result and len(result) == 1:
                print(f"DB Update: {result[0]}")

                # Update internal state - invert roll and pitch for live mode
                self.yaw = int(result[0]['yaw'])
                self.pitch = int(-result[0]['pitch'])  # Inverted for live mode
                self.roll = int(result[0]['roll'])  # Inverted for live mode (was negated before)

                # Block signals to prevent triggering update_image
                self.slider_yaw.blockSignals(True)
                self.slider_pitch.blockSignals(True)
                self.slider_roll.blockSignals(True)

                # Update sliders without triggering valueChanged signal
                self.slider_yaw.setValue(self.yaw)
                self.slider_pitch.setValue(self.pitch)
                self.slider_roll.setValue(self.roll)

                # Unblock signals
                self.slider_yaw.blockSignals(False)
                self.slider_pitch.blockSignals(False)
                self.slider_roll.blockSignals(False)

                # Update GL widget directly
                self.gl_widget.yaw = self.yaw
                self.gl_widget.pitch = self.pitch
                self.gl_widget.roll = self.roll
                self.gl_widget.update()
        except Exception as e:
            print(e)
            # Silently handle database errors to avoid spam
            if self.is_live_data_mode:
                print(f"DB fetch error (will retry): {e}")

    def open_file_dialog(self):
        """Opens a file dialog to select the US image path."""
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ultrasound Clip Image",
            "",
            file_filter
        )
        if file_path:
            self.us_image_path_edit.setText(file_path)

    def load_us_image(self):
        """Loads the US image from the path in the QLineEdit."""
        path = self.us_image_path_edit.text()
        if path:
            self.gl_widget.load_us_texture(path)
        else:
            print("Please provide a file path for the US image.")

    def ui_value_reset(self):
        """Resets all transformation values and updates the UI/GL widget."""
        self.z_index = 0
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.shift_x = 0
        self.shift_y = 0
        self.is_slicing_enabled = False
        self.is_live_data_mode = False  # Reset to Manual Mode

        self.slider_z.setValue(self.z_index)
        self.slider_yaw.setValue(0)
        self.slider_pitch.setValue(0)
        self.slider_roll.setValue(0)
        self.slider_x.setValue(0)
        self.slider_y.setValue(0)

        self.slicing_checkbox.setChecked(False)
        self.live_data_checkbox.setChecked(False)  # Ensures manual mode is active

        # Clear US texture
        self.gl_widget.makeCurrent()  # Ensure GL context is active for glDeleteTextures
        if self.gl_widget.us_texture_id != 0:
            glDeleteTextures([self.gl_widget.us_texture_id])
            self.gl_widget.us_texture_id = 0
            self.us_image_path_edit.setText("")

        self.gl_widget.update()

    def update_slicing_state(self, state):
        """Toggles the slicing plane visualization state."""
        self.is_slicing_enabled = (state == Qt.CheckState.Checked.value)
        self.gl_widget.is_slicing_enabled = self.is_slicing_enabled
        self.gl_widget.update()

    def add_slider(self, name, min_val, max_val, initial_val, step):
        """Creates a labeled slider with current value display."""
        slider_container = QWidget()
        slider_container.setStyleSheet("background-color: #f7f9fb; border-radius: 4px; padding: 3px;")
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(4, 2, 4, 2)
        slider_layout.setSpacing(8)

        # 1. Name Label
        label = QLabel(name, self)
        label.setFixedWidth(130)
        label.setStyleSheet("font-weight: 500; color: #2c3e50; font-size: 12px;")
        slider_layout.addWidget(label)

        # 2. Slider - Now using ClickableSlider
        slider = ClickableSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial_val)
        slider.setTickInterval(step)
        slider.setSingleStep(step)
        slider.setMinimumHeight(20)
        slider.valueChanged.connect(self.update_image)
        slider_layout.addWidget(slider)

        # 3. Current Value Label
        label_current = QLabel(str(slider.value()), self)
        label_current.setMinimumWidth(35)
        label_current.setMaximumWidth(45)
        label_current.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        label_current.setStyleSheet(
            "color: white; font-weight: bold; font-size: 12px; background-color: #e74c3c; border-radius: 3px; padding: 2px;")
        slider.valueChanged.connect(lambda value: label_current.setText(str(value)))
        slider_layout.addWidget(label_current)

        return slider_container, slider

    def update_image(self):
        """Updates the GL widget with new transformation values from sliders."""

        # NOTE: Rotation values (yaw, pitch, roll) are only taken from the sliders
        # if the system is in Manual Mode (or if the DB is failing to update, but the main toggle handles that).
        # Since the slider valueChanged signal is always emitted, this function will execute.
        # However, the values are only updated here from the slider. In Live mode, the slider
        # is disabled and its value is constantly set by fetch_orientation_from_db.

        self.yaw = self.slider_yaw.value()
        self.pitch = self.slider_pitch.value()
        self.roll = self.slider_roll.value()

        # Shift and slice are ALWAYS manually controlled
        self.shift_x = self.slider_x.value()
        self.shift_y = self.slider_y.value()
        self.z_index = self.slider_z.value()

        # Pass updated values to the OpenGL widget
        self.gl_widget.pitch = self.pitch
        self.gl_widget.yaw = self.yaw
        self.gl_widget.roll = self.roll
        self.gl_widget.shift_x = self.shift_x
        self.gl_widget.shift_y = self.shift_y
        self.gl_widget.z_index = self.z_index

        self.gl_widget.update()

    def closeEvent(self, event: QCloseEvent):
        """Clean up OpenGL resources and BLE connection when the window is closed."""
        # Stop BLE worker thread
        if hasattr(self, 'ble_worker'):
            print("Stopping BLE worker...")
            self.ble_worker.stop()
            self.ble_worker.wait(5000)  # Wait up to 5 seconds for thread to finish

        # Stop database timer
        if hasattr(self, 'db_timer'):
            self.db_timer.stop()

        # Clean up OpenGL resources
        self.gl_widget.makeCurrent()
        if self.gl_widget.us_texture_id != 0:
            glDeleteTextures([self.gl_widget.us_texture_id])

        event.accept()


if __name__ == "__main__":

    # result = mysql_interface.db_select(None, 'SELECT roll, pitch, yaw FROM ble_receiver.probe_imu where idx = 1',
    #                                    [])
    #
    # print(result)

    app = QApplication(sys.argv)
    viewer = UltrasoundViewer()
    viewer.show()
    sys.exit(app.exec())


