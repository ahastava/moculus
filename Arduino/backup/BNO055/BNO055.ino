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


  // Serial.print("Orientation: ");
  // Serial.print(event.orientation.x);
  // Serial.print(", ");
  // Serial.print(event.orientation.y);
  // Serial.print(", ");
  // Serial.println(event.orientation.z);

  // i don't know why the default orienation is wrong, Roll and Yaw should be switched?
  // Serial.print("Orientation (Euler): Roll=");
  // Serial.print(event.orientation.roll);
  // Serial.print(" Pitch=");
  // Serial.print(event.orientation.pitch);
  // Serial.print(" Yaw=");
  // Serial.print(event.orientation.heading);
  // Serial.println(" degrees");

  Serial.print("Orientation (Euler): Yaw=");
  Serial.print(event.orientation.roll);
  Serial.print(" Pitch=");
  Serial.print(event.orientation.pitch);
  Serial.print(" Roll=");
  Serial.print(event.orientation.heading);
  Serial.println(" degrees");

// // Acceleration (linear, motion only)
//   imu::Vector<3> linearAccel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // VECTOR_ACCELEROMETER VECTOR_LINEARACCEL (no gravity)
//   Serial.print("Linear Accel (m/s^2): X=");
//   Serial.print(linearAccel.x());
//   Serial.print(" Y=");
//   Serial.print(linearAccel.y());
//   Serial.print(" Z=");
//   Serial.println(linearAccel.z());

//     // Get gyroscope data (in degrees per second)
//   imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
//   Serial.print("Gyroscope (deg/s): X=");
//   Serial.print(gyro.x());
//   Serial.print(" Y=");
//   Serial.print(gyro.y());
//   Serial.print(" Z=");
//   Serial.println(gyro.z());

//   // Get magnetometer data (in microteslas)
//   imu::Vector<3> magnetometer = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
//   Serial.print("Magnetometer (µT): X=");
//   Serial.print(magnetometer.x());
//   Serial.print(" Y=");
//   Serial.print(magnetometer.y());
//   Serial.print(" Z=");
//   Serial.println(magnetometer.z());


    imu::Quaternion quat = bno.getQuat();
    Serial.print("Quaternion: X=");
    Serial.print(quat.x());
      Serial.print(" y=");
    Serial.print(quat.y());
    Serial.print(" Z=");
    Serial.print(quat.z());
    Serial.print(" W=");
    Serial.println(quat.w());

    imu::Vector<3> eul = quat.toEuler();

    double roll = eul.x()*180.0 / M_PI;
    double pitch = eul.y()*180.0 / M_PI;
    double yaw = eul.z()*180.0 / M_PI;

    Serial.print("Eular: roll=");
    Serial.print(roll);
      Serial.print(" pitch=");
    Serial.print(pitch);
    Serial.print(" yaw=");
    Serial.println(yaw);


    float number = 12.3456;

    int scaled = (int)(number * 10 + 0.5);  // round to 1 decimal place → 123
    
    int aa = 1111;
    int bb = 2222;
    int cc = 3333;
    char hexStr[5];  // 4 digits + null terminator
    sprintf(hexStr, "%04d", aa);  // Convert to hex with padding
    
    printf("\nHex string: %s\n", hexStr);
    
    char hexStr1[5]= "1234";
    printf("\nHex string: %s\n", hexStr1);
    
    char final[16]= "000000000000000"; // 15 character for 16 bytes
    printf("\nfinal string: %s\n", final); 
    
    sprintf(final, "%04d%04d%04d000", aa, bb, cc);
        
    printf("\nfinal string: %s\n", final); 
    Serial.println(final);

  delay(500);
}