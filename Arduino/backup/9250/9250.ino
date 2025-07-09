#include <MPU9250_asukiaaa.h>
#include <ReefwingAHRS.h>

ReefwingAHRS ahrs;
SensorData data;


MPU9250_asukiaaa mySensor;
float aX, aY, aZ, aSqrt, gX, gY, gZ, mDirection, mX, mY, mZ;
#define CALIB_SEC 10

float magX_offset;
float magY_offset;
float magZ_offset;

float magX_scale;
float magY_scale;
float magZ_scale;

float avg_scale;

void setMagMinMaxAndSetOffset(MPU9250_asukiaaa* sensor, int seconds) {
  unsigned long calibStartAt = millis();
  float magX, magXMin, magXMax, magY, magYMin, magYMax, magZ, magZMin, magZMax;

  sensor->magUpdate();
  magXMin = magXMax = sensor->magX();
  magYMin = magYMax = sensor->magY();
  magZMin = magZMax = sensor->magZ();

  while(millis() - calibStartAt < (unsigned long) seconds * 1000) {
    delay(100);
    sensor->magUpdate();
    magX = sensor->magX();
    magY = sensor->magY();
    magZ = sensor->magZ();
    if (magX > magXMax) magXMax = magX;
    if (magY > magYMax) magYMax = magY;
    if (magZ > magZMax) magZMax = magZ;
    if (magX < magXMin) magXMin = magX;
    if (magY < magYMin) magYMin = magY;
    if (magZ < magZMin) magZMin = magZ;
  }

  Serial.println("before");
  Serial.println("mySensor.magXOffset = " + String(mySensor.magXOffset) + ";");
  Serial.println("mySensor.maxYOffset = " + String(mySensor.magYOffset) + ";");
  Serial.println("mySensor.magZOffset = " + String(mySensor.magZOffset) + ";");



  sensor->magXOffset = - (magXMax + magXMin) / 2;
  sensor->magYOffset = - (magYMax + magYMin) / 2;
  sensor->magZOffset = - (magZMax + magZMin) / 2;

//  magX_offset = (magXMax + magXMin) / 2;
//  magY_offset = (magYMax + magYMin) / 2;
//  magZ_offset = (magZMax + magZMin) / 2;

  Serial.println("after");
  Serial.println("mySensor.magXOffset = " + String(mySensor.magXOffset) + ";");
  Serial.println("mySensor.maxYOffset = " + String(mySensor.magYOffset) + ";");
  Serial.println("mySensor.magZOffset = " + String(mySensor.magZOffset) + ";");

  // magX_scale = (magXMax - magXMin) / 2;
  // magY_scale = (magYMax - magYMin) / 2;
  // magZ_scale = (magZMax - magZMin) / 2;
  
  // avg_scale = (magX_scale + magY_scale + magZ_scale) / 3.0;


}

void setup() {

  ahrs.begin();
  
  ahrs.setFusionAlgorithm(SensorFusion::MADGWICK);
 ahrs.setDeclination(-12.46); //-12.46 Hicksville, NY


  Serial.begin(115200);
  while (!Serial);

  Wire.begin();
  mySensor.setWire(&Wire);
  mySensor.beginAccel();
  mySensor.beginGyro();
  mySensor.beginMag();

  Serial.println("MPU9250 initialized");


  Serial.println("Start scanning values of magnetometer to get offset values.");
  Serial.println("Rotate your device for " + String(CALIB_SEC) + " seconds.");
  setMagMinMaxAndSetOffset(&mySensor, CALIB_SEC);
  Serial.println("Finished setting offset values.");




}

void loop() {
  mySensor.accelUpdate();
  mySensor.gyroUpdate();
  mySensor.magUpdate();

  // Serial.print("Accel X: ");
  // Serial.print(mySensor.accelX());
  // Serial.print(", Y: ");
  // Serial.print(mySensor.accelY());
  // Serial.print(", Z: ");
  // Serial.println(mySensor.accelZ());

  delay(500);

    aX = mySensor.accelX();
    aY = mySensor.accelY();
    aZ = mySensor.accelZ();
    // Serial.println("accelX: " + String(aX));
    // Serial.println("accelY: " + String(aY));
    // Serial.println("accelZ: " + String(aZ));

    gX = mySensor.gyroX();
    gY = mySensor.gyroY();
    gZ = mySensor.gyroZ();
  // Serial.println("gyroX: " + String(gX));
  //   Serial.println("gyroY: " + String(gY));
  //   Serial.println("gyroZ: " + String(gZ));

    mX = mySensor.magX();
    mY = mySensor.magY();
    mZ = mySensor.magZ();


  // Serial.println("magX: " + String(mX));
  //   Serial.println("maxY: " + String(mY));
  //   Serial.println("magZ: " + String(mZ));

  data.ax = aX;
  data.ay = aY;
  data.az = aZ;

  data.gx = gX;
  data.gy = gY;
  data.gz = gZ;

  data.mx = mX;
  data.my = mY;
  data.mz = mZ;

  // data.mx = (mX - magX_offset) * (avg_scale / magX_scale);
  // data.my = (mY - magY_offset) * (avg_scale / magY_scale);
  // data.mz = (mZ - magZ_offset) * (avg_scale / magZ_scale);



  ahrs.setData(data);
  ahrs.update();

  //if (millis() - previousMillis >= displayPeriod) {
    //  Display sensor data every displayPeriod, non-blocking.
    Serial.print("--> Roll: ");
    Serial.print(ahrs.angles.roll, 2);
    Serial.print("\tPitch: ");
    Serial.print(ahrs.angles.pitch, 2);
    Serial.print("\tYaw: ");
    Serial.print(ahrs.angles.yaw, 2);
    Serial.print("\tHeading: ");
    Serial.print(ahrs.angles.heading, 2);
   // Serial.print("\tLoop Frequency: ");
    //Serial.print(loopFrequency);
    //Serial.println(" Hz");
Serial.println("");
   // loopFrequency = 0;
  //  previousMillis = millis();
  //}

 // loopFrequency++;

}