//Tutorial from 
//https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085?view=all

#include <Arduino.h>

// This demo explores two reports (SH2_ARVR_STABILIZED_RV and SH2_GYRO_INTEGRATED_RV) both can be used to give 
// quartenion and euler (yaw, pitch roll) angles.  Toggle the FAST_MODE define to see other report.  
// Note sensorValue.status gives calibration accuracy (which improves over time)
#include <Adafruit_BNO08x.h>

// For SPI mode, we need a CS pin
#define BNO08X_CS 10
#define BNO08X_INT 9


// #define FAST_MODE

// For SPI mode, we also need a RESET 
//#define BNO08X_RESET 5
// but not for I2C or UART
#define BNO08X_RESET -1

struct euler_t {
  float yaw;
  float pitch;
  float roll;
} ypr;

Adafruit_BNO08x  bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

#ifdef FAST_MODE
  // Top frequency is reported to be 1000Hz (but freq is somewhat variable)
  sh2_SensorId_t reportType = SH2_GYRO_INTEGRATED_RV;
  long reportIntervalUs = 2000;
#else
  // Top frequency is about 250Hz but this report is more accurate
  sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV;
  long reportIntervalUs = 5000;
#endif

//When both reports are set to the same interval (5000 µs), they try to be read at the same time, causing:
//Polling conflicts /  I2C bottleneck / Buffer/scheduling issues - the
//slowing the report frequency from 200 to 50 every second
//by decrease it not much to 8250, we can get 110 report every second
//by decrease it around 18250, we can get ~190 report back again, but this high refresh frequency that is not needed

long accelReportIntervalUs = 8250;  //5000 us = 5ms  => 200 Hz

void setReports(sh2_SensorId_t reportType, long report_interval) {
  Serial.println("Setting desired reports");
  if (! bno08x.enableReport(reportType, report_interval)) {
    Serial.println("Could not enable stabilized remote vector");
  }

  if (!bno08x.enableReport(SH2_ACCELEROMETER, accelReportIntervalUs)) {
    Serial.println("Could not enable accelerometer report");
  }
}

// Global to store last accelerometer reading
float lastAccelX = 0;
float lastAccelY = 0;
float lastAccelZ = 0;

int lastTime_loopCount = 0;
int loopCount = 0;

int angleReportLoopCount = 0;
int BLECounter = 0;
int reportSecond = 0;

void setup(void) {

  Serial.begin(115200);
  while (!Serial) delay(10);     // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit BNO08x test!");

  delay(200);  // 200ms is usually enough to boot up BNO08x

 // pinMode(BNO08X_INT, INPUT_PULLUP);  // Configure INT pin

  // Try to initialize!
  if (!bno08x.begin_I2C()) {
  //if (!bno08x.begin_UART(&Serial1)) {  // Requires a device with > 300 byte UART buffer!
  //if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }


    
    // if (!bno08x.begin_I2C(BNO08x_I2CADDR_DEFAULT, &Wire, 0)) {
    //     Serial.println("Failed to find BNO08x chip");
    //     while (1) { delay(10); }
    // }

  Serial.println("BNO08x Found!");

  setReports(reportType, reportIntervalUs);

  Serial.println("Reading events");
  delay(100);

}

void quaternionToEuler(float qr, float qi, float qj, float qk, euler_t* ypr, bool degrees = false) {

    float sqr = sq(qr);
    float sqi = sq(qi);
    float sqj = sq(qj);
    float sqk = sq(qk);

    ypr->yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
    ypr->pitch = asin(-2.0 * (qi * qk - qj * qr) / (sqi + sqj + sqk + sqr));
    ypr->roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));

    if (degrees) {
      ypr->yaw *= RAD_TO_DEG;
      ypr->pitch *= RAD_TO_DEG;
      ypr->roll *= RAD_TO_DEG;
    }
}

void quaternionToEulerRV(sh2_RotationVectorWAcc_t* rotational_vector, euler_t* ypr, bool degrees = false) {
    quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}

void quaternionToEulerGI(sh2_GyroIntegratedRV_t* rotational_vector, euler_t* ypr, bool degrees = false) {
    quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}


void loop() {

  
//loop speed
//nothing: Loops per second: 84245
// only  delay(20); Loops per second: 50 => right
// only  delay(100); Loops per second: 10 => right

  //get the function call speed
  loopCount++;

  unsigned long now = millis();
  if (now - lastTime_loopCount >= 1000) {  // every 1 second
    Serial.print("Loops per second: ");
    Serial.print(loopCount);

    Serial.print(" reportType per second: ");
    Serial.println(angleReportLoopCount);

    loopCount = 0;          // reset counter
    angleReportLoopCount = 0;
    reportSecond += 1;
    lastTime_loopCount = now;         // reset timer
  }
  

  if (bno08x.wasReset()) { //=> Loops per second: 80389 looks ok (a little slower than nothing)
    Serial.print("sensor was reset ");
    setReports(reportType, reportIntervalUs);
  }

    //Function bno08x.getSensorEvent took:  949us => 1ms without any event registered => //Loops per second: 876 => ~1000hz
    //Function bno08x.getSensorEvent took: 3628 us => 3ms with SH2_ARVR_STABILIZED_RV report (50ms) => making the entire loop 260 per second
    //Function bno08x.getSensorEvent took: 6357 us=> 6ms with SH2_ARVR_STABILIZED_RV+ SH2_ACCELEROMETER Loops per second: 133 
    //Enabled reports	getSensorEvent() time	Approx loop freq
    //None	1 ms	~1000 Hz
    //ARVR only	3 ms	~333 Hz
    //ARVR + ACCEL	6 ms	~166 Hz

    // all function is measured using:
    // unsigned long startTime = micros();
    // // some function calls
    // unsigned long endTime = micros();
    // unsigned long elapsed = endTime - startTime;
    // Serial.print("Function bno08x.getSensorEvent took: ");
    // Serial.print(elapsed);
    // Serial.println(" us");  // microseconds



  bool re =   bno08x.getSensorEvent(&sensorValue);

  //Has to wait 1ms here! otherwise will only see 1-4 updates every second, since the I2C take time to reponse, without delay it will throttle itself.
  //The Root Cause: I2C/SPI Bus Timing Issue
  //Claude: when blindly poll the I2C bus, which causes severe blocking. The sensor doesn't know you've read the data, and you don't know when new data is ready.
  delay(1);//1 ms //essential wait



  if (re) { 
    // in this demo only one report type will be received depending on FAST_MODE define (above)

    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        // Update global accelerometer values
        lastAccelX = sensorValue.un.accelerometer.x;
        lastAccelY = sensorValue.un.accelerometer.y;
        lastAccelZ = sensorValue.un.accelerometer.z;
    //    Serial.println("SH2_ACCELEROMETER");
        break;
      case SH2_ARVR_STABILIZED_RV:

        quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true); //this function call take 230us=> 0.23 ms should be ok

        // Serial.println("SH2_ARVR_STABILIZED_RV");

        angleReportLoopCount += 1;
  
       
       if (angleReportLoopCount % 3 == 0) {  // Only print every 3rd sample (so we have ~10 data points in 1 second)
    
             char buffer[128];
          
              BLECounter++;
              float qr = sensorValue.un.arvrStabilizedRV.real;
              float qi = sensorValue.un.arvrStabilizedRV.i;
              float qj = sensorValue.un.arvrStabilizedRV.j;
              float qk = sensorValue.un.arvrStabilizedRV.k;
          
              snprintf(buffer, sizeof(buffer),
                  "%6d,%6d,%6d,%6d,"       // BLECounter, reportCount, calibration status
                  "YPR=%+6.2f,%+6.2f,%+6.2f,"  // yaw, pitch, roll (deg)
                  "Q=%+6.3f,%+6.3f,%+6.3f,%+6.3f," // quaternion (real,i,j,k)
                  "A=%+6.2f,%+6.2f,%+6.2f",    // accelerometer X,Y,Z
                  BLECounter,
                  reportSecond,
                  angleReportLoopCount,
                  sensorValue.status,
                  ypr.yaw, ypr.pitch, ypr.roll,
                  qr, qi, qj, qk,
                  lastAccelX, lastAccelY, lastAccelZ
                ); // this is 500 us => 0.5ms 
                  
        

            Serial.println(buffer);     // this is 300 us=>0.3 ms | snprintf + serial.print => total take 800 us => 0.8ms => much faster than the 7x serial print 3.7 ms
         // this is 300 us=>0.3 ms | snprintf + serial.print => total take 800 us => 0.8ms => much faster than the 7x serial print 3.7 ms
       }
        break;
      case SH2_GYRO_INTEGRATED_RV:
        quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true);
      //    Serial.println("SH2_GYRO_INTEGRATED_RV");
        break;
    }

   
  }

}



