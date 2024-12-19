//#include <Arduino_LSM6DS3.h>
#include <Arduino_LSM9DS1.h>
#include <MadgwickAHRS.h>
#include <ReefwingAHRS.h>

LSM9DS1 imu;
EulerAngles angles;

const long displayPeriod = 150; // milliseconds. Idea here is to at least get 6-7 readings sent per sec for now. Keeping it modest atm so that high throughput doesn't corrupt data
unsigned long previousMillis = 0;


void setup() {
  Serial1.begin(38400);
  Serial.begin(38400);
  // Initialise the LSM9DS1 IMU
  imu.begin();

  //  Positive magnetic declination - Kings Park, NY
  imu.setDeclination(-7.96);
  imu.setFusionAlgorithm(SensorFusion::MAHONY);
  imu.setFusionPeriod(0.01f);   // Estimated sample period = 0.01 s = 100 Hz

  //  Paste your calibration bias offset HERE
  //  This information comes from the testAndCalibrate.ino 
  //  sketch in the library examples sub-directory.

  imu.loadAccBias(0.033997, -0.084167, -0.010986);
  imu.loadGyroBias(-0.164490, -3.364563, -2.340240);
  imu.loadMagBias(0.145020, 0.011108, -0.030151);

  //  This sketch assumes that the LSM9DS1 is already calibrated, 
  //  If so, start processing IMU data. If not, run the testAndCalibrate 
  //  sketch first.
  imu.start();
}

void loop() {
  
  delay(6.5);
  angles = imu.update();

  //  Display sensor data every displayPeriod, non-blocking.
  if (millis() - previousMillis >= displayPeriod) {

    //  Uncomment to DEBUG raw sensor data:
    //  SensorData data = imu.rawData();
    //  Serial.print("ax = "); Serial.print(1000*data.ax);  
    //  Serial.print(" ay = "); Serial.print(1000*data.ay); 
    //  Serial.print(" az = "); Serial.print(1000*data.az); Serial.println(" mg");
    //  Serial.print("gx = "); Serial.print( data.gx, 2); 
    //  Serial.print(" gy = "); Serial.print( data.gy, 2); 
    //  Serial.print(" gz = "); Serial.print( data.gz, 2); Serial.println(" deg/s");
    //  Serial.print("mx = "); Serial.print(1000*data.mx ); 
    //  Serial.print(" my = "); Serial.print(1000*data.my ); 
    //  Serial.print(" mz = "); Serial.print(1000*data.mz ); Serial.println(" mG");
         
    Serial.print("AA");
    Serial.print(angles.yaw);
    Serial.print("_");
    Serial.print(angles.pitch);
    Serial.print("_");
    Serial.print(angles.roll); 
    Serial.println("ZZ");
   
    Serial1.print("AA");
    Serial1.print(angles.yaw);
    Serial1.print("_");
    Serial1.print(angles.pitch);
    Serial1.print("_");
    Serial1.print(angles.roll); 
    Serial1.println("ZZ");

    previousMillis = millis();
  }
}