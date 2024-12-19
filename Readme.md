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

#### Blueotooth setup:

The UNO is the only one that needs to be connected to the PC, and the other two just need power (can be connected to anything that powers them). 

There are two pieces of hardware on the aggregator board with the blue cases -- each of these are the receivers for the probe and background unit, respectively). The bluetooth is paired if both transmitterand the receiver's red LED is continuously creating two "blinks" followed by a pause. However, if a device is not paired, it will continuously blink at a constant rate, or if it's not powered, there will be no light.

To fix the unpaired device: sometimes even if everything is wired up correctly, the device won't pair or power up, and this can usually be fixed by just pulling the bluetooth module out of the aggregator and re-inserting it. If that doesn't work, Try disconnecting the aggregator's USB power supply from the computer and reconnecting.

If there are still issues, it's possible something got disconnected, either in the aggregator or the probe/background (Whichever pair is having the issue).