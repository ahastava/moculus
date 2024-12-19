// My Arduino IDE 2.3.3
// install and use ArdurinoBLE for Arduino Nano 33 BLE library //1.3.7 installed
#include <Arduino_LSM9DS1.h> //1.1.1 installed, old 3rd party 2.0.0 also works
#include <MadgwickAHRS.h> //1.2.0 installed, old one is 1.2.0 also
#include <ReefwingAHRS.h> //2.1.0 installed, old one is 2.1.0,  2.2.0 or 2.3.0 won't work

LSM9DS1 imu;
EulerAngles angles;

const long displayPeriod = 150; // milliseconds. Idea here is to at least get 6-7 readings sent per sec for now. Keeping it modest atm so that high throughput doesn't corrupt data
unsigned long previousMillis = 0;


int count = 0;
float rollCalib=0, pitchCalib=0, yawCalib=0;
float rollDrift = 0, pitchDrift = 0, yawDrift = 0;
float roll=0, pitch=0, heading=0;


void setup() {
   Serial1.begin(38400);
  Serial.begin(38400);
  // Initialise the LSM9DS1 IMU
  imu.begin();

  //  Positive magnetic declination - Kings Park, NY
  imu.setDeclination(-7.96);
  imu.setFusionAlgorithm(SensorFusion::MAHONY);

    //  Paste your calibration bias offset HERE
    //  This information comes from the testAndCalibrate.ino 
    //  sketch in the library examples sub-directory.

    imu.loadAccBias(-0.045776, -0.044739, -0.012573);
  	imu.loadGyroBias(2.429962, 0.358887, 1.899109);
	  imu.loadMagBias(0.236572, 0.083618, -0.062744);

    //  This sketch assumes that the LSM9DS1 is already calibrated, 
    //  If so, start processing IMU data. If not, run the testAndCalibrate 
    //  sketch first.
    imu.start();
}


void loop() {
  
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
         
    Serial.print("BB");
    Serial.print(angles.yaw);
    Serial.print("_");
    Serial.print(angles.pitch);
    Serial.print("_");
    Serial.print(angles.roll); 
    Serial.println("YY");
    
    // BB prepend at the beginnind and Y at the end: this signifies that the string is coming from the probe and not from the BAMSS
    // ** The string gets malformed due to its length. Smaller strings have a smaller chance of getting malformed. TODO Split the sends into separate
    // chunks, i.e., yaw, pitch roll for each and send them separately, where each of the three has its own signifier. 
    Serial1.print("BB");
    Serial1.print(angles.yaw);  
    Serial1.print("_");
    Serial1.print(angles.pitch);
    Serial1.print("_");
    Serial1.print(angles.roll);
    Serial1.println("Y");

    previousMillis = millis();
  }
}