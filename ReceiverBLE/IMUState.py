class IMUState:
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.yaw_delta = 0.0
        self.yaw_calibrated = 0.0

        self.quat_r = 0.0
        self.quat_i = 0.0
        self.quat_j = 0.0
        self.quat_k = 0.0

    def calibrate_yaw(self):
        self.yaw_delta = -self.yaw

    def update(self, roll, pitch, yaw, qr=None, qi=None, qj=None, qk=None):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.yaw_calibrated = self.wrap_deg_180(yaw + self.yaw_delta)

        if qr is not None:
            self.quat_r = qr
            self.quat_i = qi
            self.quat_j = qj
            self.quat_k = qk

    @staticmethod
    def wrap_deg_180(angle):
        return (angle + 180) % 360 - 180