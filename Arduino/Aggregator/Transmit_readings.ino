
#include <SoftwareSerial.h>
SoftwareSerial mySerial(8, 9); // RX, TX

//Mux control pins for analog signal (SIG_pin) default for arduino mini pro
const byte s0 = A4;
const byte s1 = A3;
const byte s2 = A2;
const byte s3 = A1;

//Mux control pins for Output signal (OUT_pin) default for arduino mini pro
const byte w0 = 6; 
const byte w1 = 5;
const byte w2 = 4;
const byte w3 = 3;

//Mux in "SIG" pin default for arduino mini pro 
const byte SIG_pin = 0; 

//Mux out "SIG" pin default for arduino mini pro
const byte OUT_pin = 2;

//Row and Column pins default for arduino mini pro
//const byte STATUS_pin = 8;
const byte COL_pin = 2;

const boolean muxChannel[16][4]={
    {0,0,0,0}, //channel 0
    {1,0,0,0}, //channel 1
    {0,1,0,0}, //channel 2
    {1,1,0,0}, //channel 3
    {0,0,1,0}, //channel 4
    {1,0,1,0}, //channel 5
    {0,1,1,0}, //channel 6
    {1,1,1,0}, //channel 7
    {0,0,0,1}, //channel 8
    {1,0,0,1}, //channel 9
    {0,1,0,1}, //channel 10
    {1,1,0,1}, //channel 11
    {0,0,1,1}, //channel 12
    {1,0,1,1}, //channel 13
    {0,1,1,1}, //channel 14
    {1,1,1,1}  //channel 15
  };


//incoming serial byte
int inByte = 0;

int valor = 0;               //variable for sending bytes to processing
int calibra[15][15];         //Calibration array for the min values of each od the 225 sensors.
int maxsensor=0;          //Variable for staring the min array
int multiplier = 254;
int pastmatrix[15][15];

String state = "z";


void setup(){
    
  pinMode(s0, OUTPUT); 
  pinMode(s1, OUTPUT); 
  pinMode(s2, OUTPUT); 
  pinMode(s3, OUTPUT); 
  
  pinMode(w0, OUTPUT); 
  pinMode(w1, OUTPUT); 
  pinMode(w2, OUTPUT); 
  pinMode(w3, OUTPUT); 
  
  pinMode(OUT_pin, OUTPUT); 
  
 // pinMode(STATUS_pin, OUTPUT);
  pinMode(COL_pin, OUTPUT);

  
  digitalWrite(s0, LOW);
  digitalWrite(s1, LOW);
  digitalWrite(s2, LOW);
  digitalWrite(s3, LOW);
  
  digitalWrite(w0, LOW);
  digitalWrite(w1, LOW);
  digitalWrite(w2, LOW);
  digitalWrite(w3, LOW);
  
  digitalWrite(OUT_pin, HIGH);
 // digitalWrite(STATUS_pin, HIGH);
  digitalWrite(COL_pin, HIGH);
  
 
  
  //Serial.begin(115200);  // dusabling this for probe baud rate 38400
  mySerial.begin(38400);
  Serial.begin(38400); // Default communication rate of the Bluetooth module

  Serial.println("\nCalibrating...\n");
    
    
  // Initialize calibration values to zero
    for(byte j = 0; j < 15; j ++){ 
      writeMux(j);
      for(byte i = 0; i < 15; i ++)
        calibra[j][i] = 0;
    }

     
  digitalWrite(COL_pin, HIGH);
}




void loop(){
      //These will store the coordinates of maximum pressure.
      int X_max = 0;
      int Y_max = 0;


      // Calibration (averages 5 values)
      for(byte k = 0; k < 5; k++){  
        for(byte j = 0; j < 15; j ++){ 
          writeMux(j);
          for(byte i = 0; i < 15; i ++)
            calibra[j][i] = calibra[j][i] + readMux(i);
        }
      } 
      
      for(byte j = 0; j < 15; j ++){ 
        writeMux(j);
        for(byte i = 0; i < 15; i ++){
          calibra[j][i] = calibra[j][i]/5;
          if(calibra[j][i] > maxsensor){
            maxsensor = calibra[j][i];
            X_max = i;
            Y_max = j;
         // Serial.print(calibra[j][i]);  // NOTE: uncomment these two lines if you want to see the whole matrix of pressure inputs
          //Serial.print("\t");
        }
      } 
      }

      
//      Serial.println();
//      Serial.print("Maximum Value: ");
//      Serial.println(maxsensor);
//      Serial.print("X = ");
//      Serial.println(X_max);
//      Serial.print("Y = ");
//      Serial.println(Y_max);
//      Serial.println();
      
     if (mySerial.available() >0)
     {
        float millisProbe2=0;
        float millisOld2 = millis();
        bool listedState=false;
        while(mySerial.available() > 0 and (millisProbe2-millisOld2)<40)
        {
          if(listedState == false)
            {
            Serial.print("state:_");
            listedState = true;
            }
          char incoming_char_background = char(mySerial.read());
          Serial.print(incoming_char_background);
          millisProbe2=millis();
       }
          millisProbe2=0;
      }

     float millisProbe = 0;
     // Regular serial is probe; mySerial is background
     if(Serial.available() > 0) { // Checks whether data is coming from the serial port
        state = "";
        bool listedState = false;
        while(Serial.available() > 0 and millisProbe<40)
        {
          if(listedState == false)
          {
            //Serial.print("state: ");
            listedState = true;
          }
          char incoming_char = char(Serial.read());

          Serial.print(incoming_char);
          if(incoming_char == 'Y' )// and state.length()>15)
          {
            Serial.println();
            Serial.print(maxsensor);
            Serial.print(" ");
            Serial.print(X_max);
            Serial.print(" ");
            Serial.println(Y_max);

            maxsensor = 0;
            X_max = 0;
            Y_max = 0;
            millisProbe+=millis();
          }
       }
       Serial.flush();
     }
     

      
     
      // Reset the maxsensor values and coordinate values. 
      maxsensor = 0;
      X_max = 0;
      Y_max = 0;
}


int readMux(byte channel){
  byte controlPin[] = {s0, s1, s2, s3};

  //loop through the 4 sig
  for(int i = 0; i < 4; i ++){
    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }

  //read the value at the SIG pin
  int val = analogRead(SIG_pin);

  //return the value
  return val;
}


void writeMux(byte channel){
  byte controlPin[] = {w0, w1, w2, w3};

  //loop through the 4 sig
  for(byte i = 0; i < 4; i ++){
    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }
}
