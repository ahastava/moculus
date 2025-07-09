import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # Works in PyQt6

class ArrowGLWidget(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(4, 4, 6, 0, 0, 0, 0, 1, 0)

        # Draw axis arrow (X+)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)  # Red
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)
        glEnd()

        glPushMatrix()
        glTranslatef(2, 0, 0)
        glRotatef(90, 0, 1, 0)
        self.draw_cone()
        glPopMatrix()

    def draw_cone(self):
        # Arrow head as cone
        quad = gluNewQuadric()
        glColor3f(1, 0, 0)
        gluCylinder(quad, 0.1, 0.0, 0.4, 20, 1)
        gluDeleteQuadric(quad)

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Arrow + QLabel")

        # QLabel setup
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid #666; background-color: black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
     #   self.image_label.setMinimumSize(300, 300)


        pixmap = QPixmap("../data/test.jpg")
        if not pixmap.isNull():
            pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Image not found")

        # OpenGL Widget
        self.gl_widget = ArrowGLWidget()
      #  self.gl_widget.setMinimumSize(400, 400)
       # self.gl_widget.setMinimumSize(400, 400)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.gl_widget)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())