#include <Arduino_BMI270_BMM150.h>

float gyroX_offset = 0, gyroY_offset = 0, gyroZ_offset = 0;

void calibrateGyro(int samples = 30) {
    float sumX = 0, sumY = 0, sumZ = 0;

    Serial.println("Calibrating gyroscope... Keep the sensor still.");

    for (int i = 0; i < samples; i++) {
        float gx, gy, gz;
        if (IMU.gyroscopeAvailable()) {
            IMU.readGyroscope(gx, gy, gz);
            sumX += gx;
            sumY += gy;
            sumZ += gz;
        }
        delay(5);  // Small delay to avoid flooding
    }

    // Compute average bias
    gyroX_offset = sumX / samples;
    gyroY_offset = sumY / samples;
    gyroZ_offset = sumZ / samples;

    Serial.print("Gyro Offsets: ");
    Serial.print(gyroX_offset); Serial.print(", ");
    Serial.print(gyroY_offset); Serial.print(", ");
    Serial.println(gyroZ_offset);

    //
    Serial.print("data.gx -= "); Serial.print(gyroX_offset);Serial.println(";");
    Serial.print("data.gy -= "); Serial.print(gyroY_offset);Serial.println(";");
    Serial.print("data.gz -= "); Serial.print(gyroZ_offset);Serial.println(";");

}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
    Serial.println("IMU initialized.");
    Serial.println("Move the sensor in all directions to collect magnetometer data.");

    calibrateGyro(30);

}

void loop() {
 
    float rawX, rawY, rawZ;

    if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(rawX, rawY, rawZ);

        // Apply calibration
        float gyroX = rawX - gyroX_offset;
        float gyroY = rawY - gyroY_offset;
        float gyroZ = rawZ - gyroZ_offset;

        Serial.print("Calibrated Gyroscope: ");
        Serial.print(gyroX); Serial.print(", ");
        Serial.print(gyroY); Serial.print(", ");
        Serial.println(gyroZ);
    }

    delay(100);

}

// Gyro Offsets: 0.15, 0.03, 0.03
// data.gx -= 0.15;
// data.gy -= 0.03;
// data.gz -= 0.03;