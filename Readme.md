# Read Me for Ultrosound project





## Steps for setting up and using the device:

#### Install Library for Arduino

// My Arduino IDE 2.3.3\
// install and use ArdurinoBLE for Arduino Nano 33 BLE library //1.3.7 installed\
#include <Arduino_LSM9DS1.h> //1.1.1 installed, old 3rd party 2.0.0 also works\
#include <MadgwickAHRS.h> //1.2.0 installed, old one is 1.2.0 also\
#include <ReefwingAHRS.h> //2.1.0 installed, old one is 2.1.0,  2.2.0 or 2.3.0 won't work


1.) Load each of the 3 .ino files into the respective Arduinos 

- The probe will get the Probe_Transmission.ino file.

- The aggregator (the large unit holding the MUXes and the Uno) will get the Transmit_readings.ino
    **Important: I believe the bluetooth module paired to the probe operates on the traditional serial TX/RX pins, so to upload and new Arduino code to the aggregator, you'll need to Unplug the RX pin in order for code publishes to go through properly and not hang.
    image.png


- The background unit (probe-like but stays stationary) will get the Background_Transmission.ino


#### To upload code to a board:

- Open the Arduino IDE. 

- Select the board in the drop-down that you want to push to
    * Probe and Background unit -> Arduino Nano 33 BLE)
    * Aggregator -> Arduino uno

- Under port, select the port that your Arduno is using:


Once everything is uploaded, you should be able to see in the serial monitor all of the serial data coming through. Be sure to close the serial monitor before running the GUI or else you'll get an "Access Denied" error when the GUI tries to read from the serial port.

2.) To run the GUI (which will open the Serial data and start outputting it to the program display on startup):

- Open up VS code within the PYQTGUI_VIP folder. Open up GUI.py and click "Run and Debug"

