// Tutorial from
// https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085?view=all

#include <Arduino.h>
// This demo explores two reports (SH2_ARVR_STABILIZED_RV and SH2_GYRO_INTEGRATED_RV) both can be used to give
// quartenion and euler (yaw, pitch roll) angles.  Toggle the FAST_MODE define to see other report.
// Note sensorValue.status gives calibration accuracy (which improves over time)
#include <Adafruit_BNO08x.h>

#include <ArduinoBLE.h>

#include <type_traits>

// For SPI mode, we need a CS pin
#define BNO08X_CS 10
#define BNO08X_INT 9

// #define FAST_MODE

// For SPI mode, we also need a RESET
// #define BNO08X_RESET 5
// but not for I2C or UART
#define BNO08X_RESET -1

struct euler_t
{
  float yaw;
  float pitch;
  float roll;
} ypr;

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

//---------------------------------------------------------

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
myPrint(const T &msg, int decimals = -1)
{
    if (Serial)
    {
        if (decimals >= 0)
            Serial.print(msg, decimals);
        else
            Serial.print(msg);
    }
}

// For all other types
template <typename T>
typename std::enable_if<!std::is_floating_point<T>::value, void>::type
myPrint(const T &msg, int decimals = -1)
{
    if (Serial)
    {
        Serial.print(msg);
    }
}

template <typename T>
void myPrintln(const T &msg)
{
    if (Serial)
    {
      Serial.println(msg);
    }
}

// void myPrintln(const T &msg, int decimals = -1)
// {
//     if (Serial)
//     {
//         if (std::is_floating_point<T>::value) // float or double
//         {
//             if (decimals >= 0)
//                 Serial.println(msg, decimals);
//             else
//                 Serial.println(msg); // default float precision
//         }
//         else
//         {
//             Serial.println(msg);     // other types
//         }
//     }
// }

//-----------------------

#ifdef FAST_MODE
// Top frequency is reported to be 1000Hz (but freq is somewhat variable)
sh2_SensorId_t reportType = SH2_GYRO_INTEGRATED_RV;
long reportIntervalUs = 2000;
#else
// Top frequency is about 250Hz but this report is more accurate
sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV;
long reportIntervalUs = 5000;
#endif

// set it differenc from reportIntervalUs to prevent I2C Polling conflicts
long accelReportIntervalUs = 8250; // 5000 us = 5ms  => 200 Hz
void setReports(sh2_SensorId_t reportType, long report_interval)
{
  myPrintln("Setting desired reports");
  if (!bno08x.enableReport(reportType, report_interval))
  {
    myPrintln("Could not enable stabilized remote vector");
  }

  // we don't need the accelerator data
  // if (!bno08x.enableReport(SH2_ACCELEROMETER, accelReportIntervalUs))
  // {
  //   myPrintln("Could not enable accelerometer report");
  // }
}

//------------------- BLE part-------------------
int BLECounter = 0;

#define BLE_UUID_ANGLE_SERVICE "9A48ECBA-2E92-082F-C079-9E75AAE428B1"

#define BLE_UUID_ACCELERATION_ALL "00000000-0000-0000-0000-0000001234DD"

BLEService angleTransferService(BLE_UUID_ANGLE_SERVICE);

BLECharacteristic allData(BLE_UUID_ACCELERATION_ALL, BLERead | BLENotify, 150);


// Global to store last accelerometer reading
// float lastAccelX = 0;
// float lastAccelY = 0;
// float lastAccelZ = 0;

int lastTime_loopCount = 0;
int loopCount = 0;

int angleReportLoopCount = 0;
int transferredLoopCount = 0;

int reportSecond= 0;
void setup(void)
{

  //--- first part
  Serial.begin(115200);
 // while (!Serial)
  //  delay(10); // will pause Zero, Leonardo, etc until serial console opens

  myPrintln("Adafruit BNO08x test!");

  delay(300); // 200ms is usually enough to boot up BNO08x

  // Try to initialize!
  if (!bno08x.begin_I2C())
  {
    // if (!bno08x.begin_UART(&Serial1)) {  // Requires a device with > 300 byte UART buffer!
    // if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    myPrintln("Failed to find BNO08x chip");
    while (1)
    {
      delay(10);
    }
  }

  myPrintln("BNO08x Found!");

  setReports(reportType, reportIntervalUs);

  myPrintln("Reading events");
  delay(100);

  // -----------BLE part ----------- -----------

  // Start BLE
  if (!BLE.begin())
  {
    myPrintln("Starting BLE failed!");
    while (1)
      ;
  }

  // Set the local name of the device
  BLE.setDeviceName("Arduino Nano 33 BLE");
  BLE.setLocalName("Arduino Nano 33 BLE");

  BLE.setAdvertisedService(angleTransferService);
  angleTransferService.addCharacteristic(allData);

  BLE.addService(angleTransferService);

  allData.writeValue("");

  // Start advertising
  BLE.advertise();

  myPrintln("BLE address:");
  myPrintln(BLE.address()); //  prints MAC as string

  myPrintln("BLE Device is ready to pair");
}

void quaternionToEuler(float qr, float qi, float qj, float qk, euler_t *ypr, bool degrees = false)
{

  float sqr = sq(qr);
  float sqi = sq(qi);
  float sqj = sq(qj);
  float sqk = sq(qk);

  ypr->yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
  ypr->pitch = asin(-2.0 * (qi * qk - qj * qr) / (sqi + sqj + sqk + sqr));
  ypr->roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));

  if (degrees)
  {
    ypr->yaw *= RAD_TO_DEG;
    ypr->pitch *= RAD_TO_DEG;
    ypr->roll *= RAD_TO_DEG;
  }
}

void quaternionToEulerRV(sh2_RotationVectorWAcc_t *rotational_vector, euler_t *ypr, bool degrees = false)
{
  quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}

void quaternionToEulerGI(sh2_GyroIntegratedRV_t *rotational_vector, euler_t *ypr, bool degrees = false)
{
  quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}

BLEDevice centralDevice; // global

void loop()
{

  // get the function call speed
  loopCount++;

  unsigned long now = millis();
  if (now - lastTime_loopCount >= 1000)
  { // every 1 second
    myPrint("Loops per second: ");
    myPrint(loopCount);

    myPrint(" angle report per second: ");
    myPrint(angleReportLoopCount);

    myPrint(" transferred per second: ");
    myPrintln(transferredLoopCount);

    loopCount = 0; // reset counter
    angleReportLoopCount = 0;
    transferredLoopCount = 0;
    reportSecond += 1;
    lastTime_loopCount = now; // reset timer
  }

  // detect new connection
  if (!centralDevice && (centralDevice = BLE.central()))
  {
    myPrint("Connected to central: ");
    myPrintln(centralDevice.address());
  }

  // handle connected central
  if (centralDevice && !centralDevice.connected())
  {
    myPrintln("Central disconnected");
    centralDevice = BLEDevice(); // reset
  }
  else if (centralDevice && centralDevice.connected())
  {
    //   myPrintln("central.connected");
    if (bno08x.wasReset())
    {
      myPrintln("sensor was reset ");
      setReports(reportType, reportIntervalUs);
    }

    bool re = bno08x.getSensorEvent(&sensorValue);

    // Has to wait 1ms here! otherwise will only see 1-4 updates every second, since the I2C take time to reponse, without delay it will throttle itself.
    // The Root Cause: I2C/SPI Bus Timing Issue
    // Claude: when blindly poll the I2C bus, which causes severe blocking. The sensor doesn't know you've read the data, and you don't know when new data is ready.
    delay(1); // 1 ms //essential wait

    if (re)
    {
      // in this demo only one report type will be received depending on FAST_MODE define (above)
      switch (sensorValue.sensorId)
      {
      case SH2_ACCELEROMETER:
        // Update global accelerometer values
        // lastAccelX = sensorValue.un.accelerometer.x;
        // lastAccelY = sensorValue.un.accelerometer.y;
        // lastAccelZ = sensorValue.un.accelerometer.z;

        break;

      case SH2_ARVR_STABILIZED_RV:
        quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true);

        angleReportLoopCount += 1;

        if (angleReportLoopCount % 7 == 0) //Change the number (e.g. 3) to get ~-10 reports every second
        { 
          
          char buffer[128];

          float qr = sensorValue.un.arvrStabilizedRV.real;
          float qi = sensorValue.un.arvrStabilizedRV.i;
          float qj = sensorValue.un.arvrStabilizedRV.j;
          float qk = sensorValue.un.arvrStabilizedRV.k;

          snprintf(buffer, sizeof(buffer),
              "%6d,%6d,%6d,%6d,%6d,"       // BLECounter, reportSecond, angleReportLoopCount, transferredLoopCount,calibration status
              "YPR=%+6.2f,%+6.2f,%+6.2f,"  // yaw, pitch, roll (deg)
              "Q=%+6.3f,%+6.3f,%+6.3f,%+6.3f", // quaternion (real,i,j,k)
           //   "A=%+6.2f,%+6.2f,%+6.2f",    // accelerometer X,Y,Z
              BLECounter,
              reportSecond,
              angleReportLoopCount,
              transferredLoopCount,
              sensorValue.status,
              ypr.yaw, ypr.pitch, ypr.roll,
              qr, qi, qj, qk
             // lastAccelX, lastAccelY, lastAccelZ
          ); // this is 500 us => 0.5ms 


          //    myPrintln(buffer);     // this is 300 us=>0.3 ms | snprintf + serial.print => total take 800 us => 0.8ms => much faster than the 7x serial print 3.7 ms
          // this is 300 us=>0.3 ms | snprintf + serial.print => total take 800 us => 0.8ms => much faster than the 7x serial print 3.7 ms
          allData.writeValue(buffer);

          transferredLoopCount += 1;
          BLECounter += 1;
          if (BLECounter > 999999)
            BLECounter = 0;
        }
        break;
      case SH2_GYRO_INTEGRATED_RV:
        // faster (more noise?)
        quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true);
        break;
      }

    }
  }
}