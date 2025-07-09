#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for serial

  if (!bno.begin()) {
    Serial.println("BNO055 not detected!");
    while (1);
  }

  delay(1000);
  bno.setExtCrystalUse(true); // Use external crystal for better accuracy
}

void loop() {
  sensors_event_t event;
  bno.getEvent(&event);


  Serial.print("Orientation (Euler): Yaw=");
  Serial.print(event.orientation.roll);
  Serial.print(" Pitch=");
  Serial.print(event.orientation.pitch);
  Serial.print(" Roll=");
  Serial.print(event.orientation.heading);
  Serial.println(" degrees");

// Acceleration (linear, motion only)
  imu::Vector<3> linearAccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  Serial.print("Linear Accel (m/s^2): X=");
  Serial.print(linearAccel.x());
  Serial.print(" Y=");
  Serial.print(linearAccel.y());
  Serial.print(" Z=");
  Serial.println(linearAccel.z());
  
  
  // Get gyroscope data (in degrees per second)
  imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  Serial.print("Gyroscope (deg/s): X=");
  Serial.print(gyro.x());
  Serial.print(" Y=");
  Serial.print(gyro.y());
  Serial.print(" Z=");
  Serial.println(gyro.z());

  // Get magnetometer data (in microteslas)
  imu::Vector<3> magnetometer = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
  Serial.print("Magnetometer (µT): X=");
  Serial.print(magnetometer.x());
  Serial.print(" Y=");
  Serial.print(magnetometer.y());
  Serial.print(" Z=");
  Serial.println(magnetometer.z());
  
  delay(500);
}