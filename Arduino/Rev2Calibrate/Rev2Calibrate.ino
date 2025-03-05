#include <Arduino_BMI270_BMM150.h>

float magX, magY, magZ;
float magX_min = 1000, magX_max = -1000;
float magY_min = 1000, magY_max = -1000;
float magZ_min = 1000, magZ_max = -1000;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
    Serial.println("IMU initialized.");
    Serial.println("Move the sensor in all directions to collect magnetometer data.");
}

void loop() {
    if (IMU.magneticFieldAvailable()) {
        IMU.readMagneticField(magX, magY, magZ);

        // Track min/max values
        if (magX < magX_min) magX_min = magX;
        if (magX > magX_max) magX_max = magX;
        if (magY < magY_min) magY_min = magY;
        if (magY > magY_max) magY_max = magY;
        if (magZ < magZ_min) magZ_min = magZ;
        if (magZ > magZ_max) magZ_max = magZ;

        // Serial.print("MagX: "); Serial.print(magX);
        // Serial.print(" MagY: "); Serial.print(magY);
        // Serial.print(" MagZ: "); Serial.println(magZ);

        Serial.print("magX_min: "); Serial.print(magX_min);
        Serial.print("magX_max: "); Serial.print(magX_max);

        Serial.print("magY_min: "); Serial.print(magY_min);
        Serial.print("magY_max: "); Serial.print(magX_max);

        Serial.print("magZ_min: "); Serial.print(magZ_min);
        Serial.print("magZ_max: "); Serial.println(magZ_max);


    }
}

//Step 2: 
// float magX_min = -74.00, magX_max = 32.00;
// float magY_min = -56.00, magY_max = 32.00;
// float magZ_min = -68.00, magZ_max = 36.00;

// float magX_offset = (magX_max + magX_min) / 2.0;  // (32 + (-74)) / 2 = (-42) / 2 = -21.0
// float magY_offset = (magY_max + magY_min) / 2.0;  // (32 + (-56)) / 2 = (-24) / 2 = -12.0
// float magZ_offset = (magZ_max + magZ_min) / 2.0;  // (36 + (-68)) / 2 = (-32) / 2 = -16.0


//Step 3:
// float magX_offset = -21.0;
// float magY_offset = -12.0;
// float magZ_offset = -16.0;
