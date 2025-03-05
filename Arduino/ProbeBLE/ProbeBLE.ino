#include <ArduinoBLE.h>

// My Arduino IDE 2.3.3
// install and use ArduinoBLE for Arduino Nano 33 BLE library //1.3.7 installed
#include <Arduino_LSM9DS1.h> //1.1.1 installed, old 3rd party 2.0.0 also works
#include <MadgwickAHRS.h> //1.2.0 installed, old one is 1.2.0 also
ingAHRS.h> //2.1.0 installed, old one is 2.1.0,  2.2.0 or 2.3.0 won't work => 2.3.5 installed, and code updated and running fine

// BLE codes defination
const long BLEDisplayFrenquency = 250;
unsigned long BLEPreviousTimestamp = millis();

float angle = 0;
int BLECounter = 0;


#define BLE_UUID_ANGLE_SERVICE               "9A48ECBA-2E92-082F-C079-9E75AAE428B1"

#define BLE_UUID_ACCELERATION_COUNTER       "123400"
#define BLE_UUID_ACCELERATION_YAW           "1234AA"
#define BLE_UUID_ACCELERATION_PITCH         "1234BB"
#define BLE_UUID_ACCELERATION_ROLL          "00000000-0000-0000-0000-0000001234CC"

BLEService angleTransferService( BLE_UUID_ANGLE_SERVICE );

BLEUnsignedLongCharacteristic  counterCharacteristic( BLE_UUID_ACCELERATION_COUNTER, BLERead | BLENotify );

BLEFloatCharacteristic yawCharacteristic( BLE_UUID_ACCELERATION_YAW, BLERead | BLENotify );
BLEFloatCharacteristic pitchCharacteristic( BLE_UUID_ACCELERATION_PITCH, BLERead | BLENotify );
BLEFloatCharacteristic rollCharacteristic( BLE_UUID_ACCELERATION_ROLL, BLERead | BLENotify );



LSM9DS1 imu;
EulerAngles angles;

int SerialCounter = 0;
const long SerialDisplayFrequency = 250;
unsigned long SerialPreviousTimestamp = millis();



float rollCalib=0, pitchCalib=0, yawCalib=0;
float rollDrift = 0, pitchDrift = 0, yawDrift = 0;
float roll=0, pitch=0, yaw=0;


void setup() {

  // Start serial communication for debugging
  Serial.begin(38400);
  while (!Serial);  // Wait for Serial to be ready

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

    BLE.addService( angleTransferService );
  // 2

  
  counterCharacteristic.writeValue( 0 );
  yawCharacteristic.writeValue( 0 );
  pitchCharacteristic.writeValue( 0 );
  rollCharacteristic.writeValue( 0 );

  // Start advertising
  BLE.advertise();

  Serial.println("BLE Device is ready to pair");

   // Initialise the LSM9DS1 IMU
  imu.begin();

  //  Positive magnetic declination - Kings Park, NY
  //Stony Brook: -12.57
  imu.setDeclination(-12.57); //-12.717  
  imu.setFusionAlgorithm(SensorFusion::MAHONY);

  //  Paste your calibration bias offset HERE
    //  This information comes from the testAndCalibrate.ino 
    //  sketch in the library examples sub-directory.
    // imu.loadAccBias(-0.045776, -0.044739, -0.012573);
  	// imu.loadGyroBias(2.429962, 0.358887, 1.899109);
	  // imu.loadMagBias(0.236572, 0.083618, -0.062744);

imu.loadAccBias(0.138428, -0.030640, -0.024963);
	imu.loadGyroBias(4.785156, 0.859833, 4.643097);
	imu.loadMagBias(0.081421, 0.051880, -0.313110);
    //  This sketch assumes that the LSM9DS1 is already calibrated, 
    //  If so, start processing IMU data. If not, run the testAndCalibrate 
    //  sketch first.
    imu.start();

}


float pitchInitial = 0, yawInitial=0, rollInitial=0; // We're going to take the first 50 readings and average as the "initial reading"
int counter = 0;
int initialTriggered = 0;

void loop() {

   angles = imu.update();
  roll = angles.roll;
  pitch = angles.pitch;
  yaw = angles.yaw;

  // //send data through serial port
  // if (millis() - SerialPreviousTimestamp >= SerialDisplayFrequency) {

  //   Serial.print("BB");
  //   Serial.print(yaw);
  //   Serial.print("_");
  //   Serial.print(pitch);
  //   Serial.print("_");
  //   Serial.print(roll); 
  //   Serial.println("YY");
    
  //   SerialPreviousTimestamp = millis();
  // }
  // SerialCounter++;

  // //send data through BLE
  BLEDevice central = BLE.central();

  if ( central )
  {
    Serial.print( "Connected to central: " );
    Serial.println( central.address() );

    while ( central.connected() )
    {
                    angles = imu.update();
  roll = angles.roll;
  pitch = angles.pitch;
  yaw = angles.yaw;

      // Check if it's time to send data (every displayPeriod, non-blocking)
      if (millis() - BLEPreviousTimestamp >= BLEDisplayFrenquency) {
        // Update the characteristic with the new counter value
            


          counterCharacteristic.writeValue( BLECounter );  
          yawCharacteristic.writeValue( yaw );
          pitchCharacteristic.writeValue( pitch );
          rollCharacteristic.writeValue( roll );
          

          // Debugging output to Serial Monitor
          Serial.print("Sending counter value over BLE:");
          Serial.print("Probe counter: ");
          Serial.print(BLECounter);

          Serial.print(" yaw:");
          Serial.print(yaw);
          Serial.print(" pitch: ");
          Serial.print(pitch);
          Serial.print(" roll: ");
          Serial.print(roll); 
          Serial.println("YY");
          
          // Update time
          BLEPreviousTimestamp = millis();
          
          // Increment the counter
          BLECounter++;
     }
    }
  }
}