import sys

import logging
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QHBoxLayout, QPushButton, QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent

import BLE_Worker
from bk import dual_imu_handler
from calibrate_worker import CalibrateWorker

import config
class IMUControlPanel(QWidget):
    """Widget for controlling a single IMU device"""

    def __init__(self, device_name, device_address, db_table_name, db_history_table_name, col_count, parent=None):
        super().__init__(parent)
        self.device_name = device_name
        self.device_address = device_address
        self.db_table_name = db_table_name
        self.db_history_table_name = db_history_table_name
        self.col_count = col_count
        self.ble_worker = None

        self.init_ui()
        self.setup_ble_worker()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Device title
        title = QLabel(f"{self.device_name} Control")
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

        # Connection status group
        status_group = QGroupBox("Connection Status")
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

        # Connection buttons
        button_layout = QHBoxLayout()
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

        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.disconnect_button)
        status_layout.addLayout(button_layout)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # IMU Data Display
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
        # --- two columns ---
        cols_layout = QHBoxLayout()

        # left column (method 1)
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("<b>Method 1</b>"))
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

        if self.col_count == 2: #col for method 2 display
        #     # right column (method 2)
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
            QScrollBar::handle:vertical
        """)
        layout.addWidget(self.log_box)

        layout.addStretch()

    def setup_ble_worker(self):
        """Setup BLE worker thread"""

        self.ble_worker = BLE_Worker.BLEWorkerThread(
            self.device_address,
            self.db_table_name,
            self.db_history_table_name
        )
        self.ble_worker.imu_data_signal.connect(self.update_imu_data)
        self.ble_worker.ble_status_signal.connect(self.update_status)
        # just use a simple binary on/off to control the connect and disconnect button
        # signal can also toggle the button, so this maybe create a flicker of the button
        # try later
        # otherwise, connect, cancel, disconnect with existing/non exist device would be too hard for both ui and ble
        # module to update
        if self.col_count == 1:
            self.ble_worker.connection_state_signal.connect(self.update_connection_state)
            self.ble_worker.start()

        self.update_count = 0
        self.refresh_rate = 0

        # Timer to update refresh Hz once per second
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_refresh_rate)
        self.refresh_timer.start(1000)  # every 1 second

    def connect_device(self):
        """Request connection to device"""
        if self.ble_worker is None:
            self.setup_ble_worker()

        self.ble_worker.request_connect()
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)

    def reset_connect_button(self):
        self.connect_button.setText("Connect")
        # self.connect_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #27ae60;
        #         color: white;
        #         border: none;
        #         padding: 8px 16px;
        #         border-radius: 5px;
        #         font-size: 13px;
        #         font-weight: 600;
        #     }
        #     QPushButton:hover { background-color: #229954; }
        # """)
        self.connect_button.clicked.disconnect()
        self.connect_button.clicked.connect(self.connect_device)

    def disconnect_device(self):

        if self.ble_worker:

            logging.info("disconnect_device")
            logging.info("self.ble_worker.is_connected: %s", self.ble_worker.is_connected)

            if not self.ble_worker.is_connected:
                self.update_status("Connection cancelled", "orange")
                self.reset_connect_button()

            self.ble_worker.request_disconnect()
            self.disconnect_button.setEnabled(False)
            self.connect_button.setEnabled(True)


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
        # self.status_label.setStyleSheet(
        #     f"QLabel {{ color: {color}; font-weight: bold; font-size: 13px; }}"
        # )

        self.status_label.setWordWrap(True)
        self.status_label.setFixedWidth(250)  # prevent sudden window expansion
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

        # --- Add to log ---
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {status_message}")
        self.log_box.moveCursor(self.log_box.textCursor().MoveOperation.End)

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
            self.disconnect_button.setEnabled(False)

    def cleanup(self):
        """Cleanup resources"""
        if self.ble_worker:
            self.ble_worker.stop()
            self.ble_worker.wait(5000)


class CalibratedIMUPanel(IMUControlPanel):
    """UI panel for calibrated IMU (same style as others but different function)"""

    def __init__(self, db_manager, parent=None):

        #device_name, device_address, db_table_name, db_history_table_name, col_count,
        super().__init__("Calibrated IMU", "N/A", "N/A", "N/A", 2, parent)


        # Create calibration worker
        self.calibrate_worker = None
        self.setup_calibrate_worker()

        # Change button labels
        self.connect_button.setText("Reset History Table")
        self.disconnect_button.setText("Start Calibration")
        self.disconnect_button.setEnabled(True)  # Enable calibration button

        # Rewire buttons
        self.connect_button.clicked.disconnect()
        self.connect_button.clicked.connect(self.reset_history_table)

        self.disconnect_button.clicked.disconnect()
        self.disconnect_button.clicked.connect(self.start_calibration)

        # Update initial status
        self.status_label.setText("Ready for calibration")
        self.connection_indicator.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #f1c40f;
            }
        """)

    def setup_calibrate_worker(self):
        """Setup calibration worker thread"""
        self.calibrate_worker = CalibrateWorker()

        # Connect signals
        self.calibrate_worker.status_signal.connect(self.update_status)
        self.calibrate_worker.calibration_result_signal.connect(self.on_calibration_complete)
        self.calibrate_worker.reset_complete_signal.connect(self.on_reset_complete)
        self.calibrate_worker.error_signal.connect(self.on_error)

        # Start the worker thread (it will wait for operations)
        self.calibrate_worker.start()
        logging.info("Calibrate worker started")

    def reset_history_table(self):
        """Reset IMU history tables"""
        self.log_box.append("[Calibrated IMU] Resetting history tables...")
        self.connect_button.setEnabled(False)

        # Request the operation
        self.calibrate_worker.request_reset()

    def start_calibration(self):
        """Toggle start/stop calibration loop"""
        if not self.calibrate_worker.continuous_mode:
            # Start continuous calibration
            self.log_box.append("[Calibrated IMU] Starting continuous calibration...")
            self.calibrate_worker.toggle_continuous()
            self.disconnect_button.setText("Stop Calibration")
            self.disconnect_button.setStyleSheet("""
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
            # Request the operation once; worker keeps looping
            self.calibrate_worker.request_calibration()
        else:
            # Stop continuous calibration
            self.log_box.append("[Calibrated IMU] Stopping calibration...")
            self.calibrate_worker.toggle_continuous()
            self.disconnect_button.setText("Start Calibration")
            self.disconnect_button.setStyleSheet("""
                    QPushButton {
                        background-color: #27ae60;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 5px;
                        font-size: 13px;
                        font-weight: 600;
                    }
                    QPushButton:hover { background-color: #229954; }
                """)

    def on_calibration_complete(self, method1_result, method2_result):
        """Handle calibration completion"""
        # Update UI with method 1 results
        self.update_imu_data(method1_result.roll, method1_result.pitch, method1_result.yaw,
                             method2_result.roll, method2_result.pitch, method2_result.yaw)

        # Log results
        # self.log_box.append(
        #     f"[Method 1] Roll={method1_result.roll:.1f}°, "
        #     f"Pitch={method1_result.pitch:.1f}°, "
        #     f"Yaw={method1_result.yaw:.1f}°"
        # )
        # self.log_box.append(
        #     f"[Method 2] Roll={method2_result.roll:.1f}°, "
        #     f"Pitch={method2_result.pitch:.1f}°, "
        #     f"Yaw={method2_result.yaw:.1f}°"
        # )

        self.disconnect_button.setEnabled(True)

    def on_reset_complete(self):
        """Handle reset completion"""
        self.log_box.append("[Calibrated IMU] History tables cleared successfully.")
        self.connect_button.setEnabled(True)

    def on_error(self, error_message):
        """Handle errors"""
        self.log_box.append(f"[ERROR] {error_message}")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(True)

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

        # Initialize database managers
        self.probe_db_manager = None
        self.car_db_manager = None
       # self.init_database_managers()

        self.init_ui()

    def init_database_managers(self):
        """Initialize database managers for both IMUs"""
        # This will be called synchronously, but the actual pool creation is async
        # You may need to modify this based on your setup
        DB_CONFIG = dual_imu_handler.DB_CONFIG

        self.probe_db_manager = dual_imu_handler.UltraFastDatabaseManager(
            DB_CONFIG, "probe_imu", "probe_imu_history"
        )

        self.car_db_manager = dual_imu_handler.UltraFastDatabaseManager(
            DB_CONFIG, "car_imu", "car_imu_history"
        )

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
            "probe_imu",
            "probe_imu_history",
            1,
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
            1,
            self
        )
        imu_layout.addWidget(self.car_panel)

        #### start of third column
        self.calibrated_panel = CalibratedIMUPanel(self.probe_db_manager, self)
        imu_layout.addWidget(self.calibrated_panel)
        # ### end of third column
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
        # Cleanup IMU panels
        self.probe_panel.cleanup()
        self.car_panel.cleanup()

        event.accept()


if __name__ == "__main__":

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # --- Console handler (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # --- File handler ---
    # log_filename = f"./log/example_{datetime.now():%Y%m%d_%H%M%S}.log"
    # file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    # file_handler.setLevel(logging.INFO)
    # file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # file_handler.setFormatter(file_formatter)

    # --- Add both handlers ---
    logger.addHandler(console_handler)
    #logger.addHandler(file_handler)

    app = QApplication(sys.argv)
    window = DualIMUController()
    window.show()
    sys.exit(app.exec())
