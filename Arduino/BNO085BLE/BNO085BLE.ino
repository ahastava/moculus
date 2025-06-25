//Tutorial from 
//https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085?view=all

#include <Arduino.h>
// This demo explores two reports (SH2_ARVR_STABILIZED_RV and SH2_GYRO_INTEGRATED_RV) both can be used to give 
// quartenion and euler (yaw, pitch roll) angles.  Toggle the FAST_MODE define to see other report.  
// Note sensorValue.status gives calibration accuracy (which improves over time)
#include <Adafruit_BNO08x.h>

#include <ArduinoBLE.h>

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
void setReports(sh2_SensorId_t reportType, long report_interval) {
  Serial.println("Setting desired reports");
  if (! bno08x.enableReport(reportType, report_interval)) {
    Serial.println("Could not enable stabilized remote vector");
  }
}


//------------------- BLE part-------------------
int BLECounter = 0;

#define BLE_UUID_ANGLE_SERVICE               "9A48ECBA-2E92-082F-C079-9E75AAE428B1"

#define BLE_UUID_ACCELERATION_COUNTER       "123400"
#define BLE_UUID_ACCELERATION_YAW           "1234AA"
#define BLE_UUID_ACCELERATION_PITCH         "1234BB"
#define BLE_UUID_ACCELERATION_ROLL          "00000000-0000-0000-0000-0000001234CC"
#define BLE_UUID_ACCELERATION_ALL          "00000000-0000-0000-0000-0000001234DD"

BLEService angleTransferService( BLE_UUID_ANGLE_SERVICE );

BLEUnsignedLongCharacteristic  counterCharacteristic( BLE_UUID_ACCELERATION_COUNTER, BLERead | BLENotify );

BLEFloatCharacteristic yawCharacteristic( BLE_UUID_ACCELERATION_YAW, BLERead | BLENotify );
BLEFloatCharacteristic pitchCharacteristic( BLE_UUID_ACCELERATION_PITCH, BLERead | BLENotify );
BLEFloatCharacteristic rollCharacteristic( BLE_UUID_ACCELERATION_ROLL, BLERead | BLENotify );
BLECharacteristic allData(BLE_UUID_ACCELERATION_ALL, BLERead | BLENotify, 50);

//---------------------------------------------------------


void setup(void) {

  Serial.begin(115200);
  while (!Serial) delay(10);     // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit BNO08x test!");

  // Try to initialize!
  if (!bno08x.begin_I2C()) {
  //if (!bno08x.begin_UART(&Serial1)) {  // Requires a device with > 300 byte UART buffer!
  //if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO08x Found!");


  setReports(reportType, reportIntervalUs);

  Serial.println("Reading events");
  delay(100);
  
  // -----------BLE part ----------- -----------

    // Start BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  // Set the local name of the device
  BLE.setDeviceName( "Arduino Nano 33 BLE" );
  BLE.setLocalName( "Arduino Nano 33 BLE" );

  // 1
    BLE.setAdvertisedService( angleTransferService );
    angleTransferService.addCharacteristic( counterCharacteristic );
    angleTransferService.addCharacteristic( yawCharacteristic );
    angleTransferService.addCharacteristic( pitchCharacteristic );
    angleTransferService.addCharacteristic( rollCharacteristic );
    angleTransferService.addCharacteristic( allData );


    BLE.addService( angleTransferService );
  // 2

  
  counterCharacteristic.writeValue( 0 );
  yawCharacteristic.writeValue( 0 );
  pitchCharacteristic.writeValue( 0 );
  rollCharacteristic.writeValue( 0 );
 allData.writeValue( "" );

  // Start advertising
  BLE.advertise();
  
  Serial.println("BLE address:");
  Serial.println(BLE.address());  //  prints MAC as string

  Serial.println("BLE Device is ready to pair");
 // ---------------------- -----------
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

  // //send data through BLE
  BLEDevice central = BLE.central();

  if ( central )
  {
    Serial.print( "Connected to central: " );
    Serial.println( central.address() );

    while ( central.connected() )
    {

        if (bno08x.wasReset()) {
          Serial.print("sensor was reset ");
          setReports(reportType, reportIntervalUs);
        }
        
        if (bno08x.getSensorEvent(&sensorValue)) {
          // in this demo only one report type will be received depending on FAST_MODE define (above)
          switch (sensorValue.sensorId) {
            case SH2_ARVR_STABILIZED_RV:
              quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true);
            case SH2_GYRO_INTEGRATED_RV:
              // faster (more noise?)
              quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true);
              break;
          }
          static long last = 0;
          long now = micros();
          Serial.print(now - last);             Serial.print("\t");
          last = now;
          Serial.print(sensorValue.status);     Serial.print("\t");  // This is accuracy in the range of 0 to 3
          Serial.print(ypr.yaw);                Serial.print("\t");
          Serial.print(ypr.pitch);              Serial.print("\t");
          Serial.println(ypr.roll);


          // Serial.print(sensorValue.un.arvrStabilizedRV.real);  Serial.print("\t");
          // Serial.print(sensorValue.un.arvrStabilizedRV.i);     Serial.print("\t");
          // Serial.print(sensorValue.un.arvrStabilizedRV.j);     Serial.print("\t");
          // Serial.println(sensorValue.un.arvrStabilizedRV.k);   



          char buffer[50];  // large enough

          // 123456,+123.45,+12.34,+12.34
          snprintf(buffer, sizeof(buffer), "%6d,%+6.2f,%+6.2f,%+6.2f", BLECounter, ypr.yaw, ypr.pitch, ypr.roll);

         // Serial.println(buffer);

          counterCharacteristic.writeValue( BLECounter );  
          yawCharacteristic.writeValue( ypr.yaw );
          pitchCharacteristic.writeValue( ypr.pitch );
          rollCharacteristic.writeValue( ypr.roll );

          allData.writeValue(buffer);
          delay(10);
          BLECounter += 1;
          if(BLECounter > 999999)
            BLECounter = 0;

        }

      }
  }

}