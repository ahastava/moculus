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


        Serial.print("magX_min: "); Serial.print(magX_min);
        Serial.print(" magX_max: "); Serial.print(magX_max);

        Serial.print(" magY_min: "); Serial.print(magY_min);
        Serial.print(" magY_max: "); Serial.print(magX_max);

        Serial.print(" magZ_min: "); Serial.print(magZ_min);
        Serial.print(" magZ_max: "); Serial.println(magZ_max);

        //Step 2: 
        float magX_offset = (magX_min + magX_max) / 2;
        float magY_offset = (magY_min + magY_max) / 2;
        float magZ_offset = (magZ_min + magZ_max) / 2;

        float magX_scale = (magX_max - magX_min) / 2;
        float magY_scale = (magY_max - magY_min) / 2;
        float magZ_scale = (magZ_max - magZ_min) / 2;


        //copy and paste this to other file:
        Serial.print("data.mx = (rawX - "); Serial.print(magX_offset); Serial.print(")/"); Serial.print(magX_scale);Serial.println(";");
        Serial.print("data.my = (rawY - "); Serial.print(magY_offset); Serial.print(")/"); Serial.print(magY_scale);Serial.println(";");
        Serial.print("data.mz = (rawZ - "); Serial.print(magZ_offset); Serial.print(")/"); Serial.print(magZ_scale);Serial.println(";");

        //sample output
        // data.mx = (rawX - magX_offset) / magX_scale;
        // data.my = (rawY - magY_offset) / magY_scale;
        // data.mz = (rawZ - magZ_offset) / magZ_scale;

    }


}


// float magX_min = -64.00, magX_max = 31.00;
// float magY_min = -56.00, magY_max = 31.00;
// float magZ_min = -60.00, magZ_max = 37.00;

// float magX_offset = (magX_max + magX_min) / 2.0;  // (32 + (-74)) / 2 = (-42) / 2 = -21.0
// float magY_offset = (magY_max + magY_min) / 2.0;  // (32 + (-56)) / 2 = (-24) / 2 = -12.0
// float magZ_offset = (magZ_max + magZ_min) / 2.0;  // (36 + (-68)) / 2 = (-32) / 2 = -16.0


//Step 3:
// float magX_offset = -21.0;
// float magY_offset = -12.0;
// float magZ_offset = -16.0;
