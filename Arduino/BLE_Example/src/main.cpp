// Simple service and characteristic - just sends "1234"

#include <ArduinoBLE.h>

#define BLE_SERVICE_UUID "9A48ECBA-2E92-082F-C079-9E75AAE428B1"

#define BLE_CHAR_UUID "00000000-0000-0000-0000-0000001234DD"

BLEService testService(BLE_SERVICE_UUID);
BLEStringCharacteristic testChar(BLE_CHAR_UUID, BLERead | BLENotify, 20);

void setup() {
  Serial.begin(115200);
  // while (!Serial);
  
  Serial.println("Starting BLE...");
  
  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }
  
  BLE.setLocalName("TestDevice");
  BLE.setDeviceName("TestDevice");
  
  BLE.setAdvertisedService(testService);
  testService.addCharacteristic(testChar);
  BLE.addService(testService);
  
  testChar.writeValue("0000");  // Initial value
  
  BLE.advertise();
  
  Serial.print("Address: ");
  Serial.println(BLE.address());
  Serial.println("Ready!");

}

BLEDevice centralDevice;
void loop() {

  BLE.poll();               // helps service BLE stack
  delay(5);                 // keep small; avoid 100ms if possible

  // delay(100);
  if (!centralDevice) {
      BLEDevice c = BLE.central();
      if (c) {
        centralDevice = c;
        Serial.print("Connected to central: ");
        Serial.println(centralDevice.address());
      }
  }

  // handle connected central
  if (centralDevice && !centralDevice.connected())
  {
    Serial.println("Central disconnected");
    centralDevice = BLEDevice(); // reset
  }
  else if (centralDevice && centralDevice.connected())
  {
    delay(100); // 1 ms //essential wait

    testChar.writeValue("1234");
    Serial.println("Sent: 1234");
  }
}