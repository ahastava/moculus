import sys
import logging
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QHBoxLayout, QPushButton, QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent

import BLE_Worker
from calibrate_worker import CalibrateWorker
from IMUState import IMUState

import config


class BaseIMUDisplayPanel(QWidget):
    """Base class for IMU display panels with common UI elements"""

    def __init__(self, panel_title, col_count, parent=None):
        super().__init__(parent)
        self.panel_title = panel_title
        self.col_count = col_count

        self.update_count = 0
        self.refresh_rate = 0

        self.init_ui()
        self.setup_refresh_timer()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Device title
        title = QLabel(f"{self.panel_title}")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
                background-color: #3498db;
                color: white;
                border-radius: 6px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Status group
        status_group = self.create_status_group()
        layout.addWidget(status_group)

        # IMU Data Display
        data_group = self.create_data_display_group()
        layout.addWidget(data_group)

        # Log display box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(100)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 6px;
            }
            QScrollBar:vertical {
                background: #2c3e50;
                width: 10px;
                margin: 0px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.log_box)
        layout.addStretch()

    def create_status_group(self):
        """Create status display group - to be customized by subclasses"""
        status_group = QGroupBox("Status")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #3498db;
            }
        """)
        status_layout = QVBoxLayout()

        # Status indicator
        indicator_layout = QHBoxLayout()
        self.connection_indicator = QLabel("●")
        self.connection_indicator.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #e74c3c;
            }
        """)
        self.connection_indicator.setFixedWidth(30)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        indicator_layout.addWidget(self.connection_indicator)
        indicator_layout.addWidget(self.status_label, stretch=1)
        status_layout.addLayout(indicator_layout)

        # Control buttons - to be added by subclasses
        self.button_layout = QHBoxLayout()
        status_layout.addLayout(self.button_layout)

        status_group.setLayout(status_layout)
        return status_group

    def create_data_display_group(self):
        """Create IMU data display group"""
        data_group = QGroupBox("IMU Data")
        data_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #3498db;
            }
        """)

        cols_layout = QHBoxLayout()

        # Column 1 - Method 1 or Primary display
        col1 = QVBoxLayout()
        col1_title = "Method 1" if self.col_count == 2 else ""
        col1.addWidget(QLabel(f"<b>{col1_title}</b>"))

        self.yaw_label_1 = QLabel("Yaw: --")
        self.pitch_label_1 = QLabel("Pitch: --")
        self.roll_label_1 = QLabel("Roll: --")

        for label in [self.yaw_label_1, self.pitch_label_1, self.roll_label_1]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    color: #2c3e50;
                    padding: 6px;
                    background-color: #ecf0f1;
                    border-radius: 4px;
                    margin: 2px;
                }
            """)
            col1.addWidget(label)

        self.refresh_label_1 = QLabel("Refresh: -- Hz")
        self.refresh_label_1.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                padding: 6px;
                background-color: #ecf0f1;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        col1.addWidget(self.refresh_label_1)
        cols_layout.addLayout(col1)

        # Column 2 - Method 2 (optional)
        if self.col_count == 2:
            col2 = QVBoxLayout()
            col2.addWidget(QLabel("<b>Method 2</b>"))

            self.yaw_label_2 = QLabel("Yaw: --")
            self.pitch_label_2 = QLabel("Pitch: --")
            self.roll_label_2 = QLabel("Roll: --")

            for lbl in [self.yaw_label_2, self.pitch_label_2, self.roll_label_2]:
                lbl.setStyleSheet("""
                    QLabel {
                        font-size: 14px;
                        font-weight: bold;
                        color: #2c3e50;
                        padding: 6px;
                        background-color: #ecf0f1;
                        border-radius: 4px;
                        margin: 2px;
                        min-width: 100px;
                    }
                """)
                col2.addWidget(lbl)

            self.refresh_label_2 = QLabel("Refresh: -- Hz")
            self.refresh_label_2.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    color: #2c3e50;
                    padding: 6px;
                    background-color: #ecf0f1;
                    border-radius: 4px;
                    margin: 2px;
                }
            """)
            col2.addWidget(self.refresh_label_2)
            cols_layout.addLayout(col2)

        data_group.setLayout(cols_layout)
        return data_group

    def setup_refresh_timer(self):
        """Setup timer for refresh rate calculation"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_refresh_rate)
        self.refresh_timer.start(1000)  # every 1 second

    def update_imu_data(self, roll, pitch, yaw, roll_2=0, pitch_2=0, yaw_2=0):
        """Update IMU data display"""
        self.roll_label_1.setText(f"Roll: {roll:.1f}°")
        self.pitch_label_1.setText(f"Pitch: {pitch:.1f}°")
        self.yaw_label_1.setText(f"Yaw: {yaw:.1f}°")

        if self.col_count == 2:
            self.roll_label_2.setText(f"Roll: {roll_2:.1f}°")
            self.pitch_label_2.setText(f"Pitch: {pitch_2:.1f}°")
            self.yaw_label_2.setText(f"Yaw: {yaw_2:.1f}°")

        self.update_count += 1

    def update_refresh_rate(self):
        """Update refresh rate display once per second"""
        self.refresh_rate = self.update_count
        self.update_count = 0
        self.refresh_label_1.setText(f"Refresh: {self.refresh_rate} Hz")

        if self.col_count == 2:
            self.refresh_label_2.setText(f"Refresh: {self.refresh_rate} Hz")

    def update_status(self, status_message, color_code):
        """Update status label"""
        self.status_label.setText(f"Status: {status_message}")

        color_map = {
            'green': "#27ae60",
            'red': "#e74c3c",
            'orange': "#f39c12",
            'yellow': "#f1c40f"
        }

        color = color_map.get(color_code.lower(), "#2c3e50")

        self.status_label.setWordWrap(True)
        self.status_label.setFixedWidth(250)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                font-weight: bold;
                color:{color};
                background-color: #ecf0f1;
                border-radius: 4px;
                padding: 4px;
            }}
        """)

        # Add to log
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {status_message}")
        self.log_box.moveCursor(self.log_box.textCursor().MoveOperation.End)

    def cleanup(self):
        """Cleanup resources - to be overridden by subclasses"""
        pass


class IMUControlPanel(BaseIMUDisplayPanel):
    """Widget for controlling a single IMU device with BLE connection"""

    def __init__(self, device_name, device_address, db_table_name, db_history_table_name, parent=None):
        self.device_name = device_name
        self.device_address = device_address
        self.db_table_name = db_table_name
        self.db_history_table_name = db_history_table_name
        self.ble_worker = None

        # Initialize base class with single column display
        super().__init__(f"{device_name} Control", 1, parent)

        # Add BLE-specific buttons
        self.add_ble_control_buttons()

        # Setup BLE worker
        self.setup_ble_worker()

    def add_ble_control_buttons(self):
        """Add BLE connection control buttons"""
        self.connect_button = QPushButton("Connect")
        self.connect_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.connect_button.clicked.connect(self.connect_device)

        self.reset_button = QPushButton("Reset Yaw")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)

        self.reset_button.clicked.connect(self.calibrate_yaw)
        self.reset_button.setEnabled(False)

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.disconnect_button.setEnabled(False)

        self.button_layout.addWidget(self.connect_button)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addWidget(self.disconnect_button)

    def setup_ble_worker(self):
        """Setup BLE worker thread"""
        self.ble_worker = BLE_Worker.BLEWorkerThread(
            self.device_address,
            self.db_table_name,
            self.db_history_table_name
        )
        self.ble_worker.imu_data_signal.connect(self.on_imu_data_received)
        self.ble_worker.ble_status_signal.connect(self.update_status)
        self.ble_worker.connection_state_signal.connect(self.update_connection_state)
        self.ble_worker.start()

    def connect_device(self):
        """Request connection to device"""
        if self.ble_worker is None:
            self.setup_ble_worker()

        self.ble_worker.request_connect()
        self.connect_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.disconnect_button.setEnabled(True)

    def reset_connect_button(self):
        """Reset connect button to initial state"""
        self.connect_button.setText("Connect")
        self.connect_button.clicked.disconnect()
        self.connect_button.clicked.connect(self.connect_device)

    def calibrate_yaw(self):
        if not self.ble_worker:
            self.update_status("Not connected (no BLE worker).", "orange")
            return
        self.ble_worker.request_yaw_reset()

    def disconnect_device(self):
        """Disconnect from device"""
        if self.ble_worker:
            logging.info("disconnect_device")
            logging.info("self.ble_worker.is_connected: %s", self.ble_worker.is_connected)

            if not self.ble_worker.is_connected:
                self.update_status("Connection cancelled", "orange")
                self.reset_connect_button()

            self.ble_worker.request_disconnect()
            self.disconnect_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.connect_button.setEnabled(True)

    def on_imu_data_received(self, imu: IMUState, roll_2=0, pitch_2=0, yaw_2=0):
        """Handle incoming IMU data from BLE worker"""

        # Update display with calibrated yaw
        self.roll_label_1.setText(f"Roll: {imu.roll:.1f}°")
        self.pitch_label_1.setText(f"Pitch: {imu.pitch:.1f}°")
        self.yaw_label_1.setText(f"Yaw: {imu.yaw:.1f}°, calibrated: {imu.yaw_calibrated:.1f}")

        self.update_count += 1

    def update_connection_state(self, is_connected):
        """Update connection state UI"""
        if is_connected:
            logging.info("is connected")
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: #27ae60;
                }
            """)
            self.connect_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.disconnect_button.setEnabled(True)
        else:
            logging.info("not connected")
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: #e74c3c;
                }
            """)
            self.connect_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.disconnect_button.setEnabled(False)

    def cleanup(self):
        """Cleanup resources"""
        if self.ble_worker:
            self.ble_worker.stop()
            self.ble_worker.wait(5000)


class CalibratedIMUPanel(BaseIMUDisplayPanel):
    """UI panel for calibrated IMU - displays results from two calibration methods"""

    def __init__(self, db_manager, parent=None):
        # Initialize base class with two column display
        super().__init__("Calibrated IMU", 2, parent)

        # Add calibration-specific buttons
        self.add_calibration_control_buttons()

        # Create calibration worker
        self.calibrate_worker = None
        self.setup_calibrate_worker()

        # Update initial status
        self.status_label.setText("Ready for calibration")
        self.connection_indicator.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #f1c40f;
            }
        """)

    def add_calibration_control_buttons(self):
        """Add calibration control buttons"""
        self.reset_history_button = QPushButton("Reset History Table")
        self.reset_history_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.reset_history_button.clicked.connect(self.reset_history_table)

        # Empty button for spacing
        self.spacer_button = QPushButton("")
        self.spacer_button.setEnabled(False)
        self.spacer_button.setVisible(False)

        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.calibrate_button.clicked.connect(self.start_calibration)

        self.button_layout.addWidget(self.reset_history_button)
        self.button_layout.addWidget(self.spacer_button)
        self.button_layout.addWidget(self.calibrate_button)

    def setup_calibrate_worker(self):
        """Setup calibration worker thread"""
        self.calibrate_worker = CalibrateWorker()

        # Connect signals
        self.calibrate_worker.status_signal.connect(self.update_status)
        self.calibrate_worker.calibration_result_signal.connect(self.on_calibration_complete)
        self.calibrate_worker.reset_complete_signal.connect(self.on_reset_complete)
        self.calibrate_worker.error_signal.connect(self.on_error)

        # Start the worker thread
        self.calibrate_worker.start()
        logging.info("Calibrate worker started")

    def reset_history_table(self):
        """Reset IMU history tables"""
        self.log_box.append("[Calibrated IMU] Resetting history tables...")
        self.reset_history_button.setEnabled(False)
        self.calibrate_worker.request_reset()

    def start_calibration(self):
        """Toggle start/stop calibration loop"""
        if not self.calibrate_worker.continuous_mode:
            # Start continuous calibration
            self.log_box.append("[Calibrated IMU] Starting continuous calibration...")
            self.calibrate_worker.toggle_continuous()
            self.calibrate_button.setText("Stop Calibration")
            self.calibrate_button.setStyleSheet("""
                QPushButton {
                    background-color: #e67e22;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover { background-color: #d35400; }
            """)
            self.calibrate_worker.request_calibration()
        else:
            # Stop continuous calibration
            self.log_box.append("[Calibrated IMU] Stopping calibration...")
            self.calibrate_worker.toggle_continuous()
            self.calibrate_button.setText("Start Calibration")
            self.calibrate_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover { background-color: #c0392b; }
            """)

    def on_calibration_complete(self, method1_result_imu, method2_result_imu):
        """Handle calibration completion"""
        # Update UI with both methods' results
        self.update_imu_data(
            method1_result_imu.roll, method1_result_imu.pitch, method1_result_imu.yaw,
            method2_result_imu.roll, method2_result_imu.pitch, method2_result_imu.yaw
        )
        self.calibrate_button.setEnabled(True)

    def on_reset_complete(self):
        """Handle reset completion"""
        self.log_box.append("[Calibrated IMU] History tables cleared successfully.")
        self.reset_history_button.setEnabled(True)

    def on_error(self, error_message):
        """Handle errors"""
        self.log_box.append(f"[ERROR] {error_message}")
        self.reset_history_button.setEnabled(True)
        self.calibrate_button.setEnabled(True)

    def cleanup(self):
        """Cleanup resources"""
        if self.calibrate_worker:
            print("Stopping calibrate worker...")
            self.calibrate_worker.stop()
            self.calibrate_worker.wait(5000)
            print("Calibrate worker stopped")


class DualIMUController(QMainWindow):
    """Main window for dual IMU control"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual IMU Controller")
        self.setGeometry(100, 100, 900, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Apply global styling
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Dual IMU Control System")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 15px;
                background-color: #3498db;
                color: white;
                border-radius: 8px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # IMU panels side by side
        imu_layout = QHBoxLayout()
        imu_layout.setSpacing(15)

        # Probe IMU Panel
        PROBE_ADDRESS = config.PROBE_ADDRESS
        self.probe_panel = IMUControlPanel(
            "Probe IMU",
            PROBE_ADDRESS,
            "probe_imu", #"probe_imu", car_imu
            "probe_imu_history", #"probe_imu_history", car_imu_history
            self
        )
        imu_layout.addWidget(self.probe_panel)

        # Car IMU Panel
        CAR_ADDRESS = config.CAR_ADDRESS
        self.car_panel = IMUControlPanel(
            "Car IMU",
            CAR_ADDRESS,
            "car_imu",
            "car_imu_history",
            self
        )
        imu_layout.addWidget(self.car_panel)

        # Calibrated IMU Panel
        self.calibrated_panel = CalibratedIMUPanel(None, self)
        imu_layout.addWidget(self.calibrated_panel)

        main_layout.addLayout(imu_layout)

        # Close button
        close_button = QPushButton("Close Application")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)

    def closeEvent(self, event: QCloseEvent):
        """Clean up resources when closing"""
        self.probe_panel.cleanup()
        self.car_panel.cleanup()
        self.calibrated_panel.cleanup()
        event.accept()


if __name__ == "__main__":
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    app = QApplication(sys.argv)
    window = DualIMUController()
    window.show()
    sys.exit(app.exec())