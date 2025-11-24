#include <Arduino.h>

// Nano 33 BLE Sense (3.3V). Scans a 16x16 Velostat pressure matrix via two CD74HC4067.
// Row MUX: selects row to excite via Rfixed from 3V3 -> measure at A0
// Col MUX: selects column to GND
// Tune Rfixed (10k–100k) for best dynamic range. Start ~33k.

const int ROW_S0 = 2;
const int ROW_S1 = 3;
const int ROW_S2 = 4;
const int ROW_S3 = 5;

const int COL_S0 = 8;
const int COL_S1 = 9;
const int COL_S2 = 10;
const int COL_S3 = 11;

// Optional EN pins if you didn’t tie to GND (comment out if EN is tied LOW)
// const int ROW_EN = 10; // LOW = enabled
// const int COL_EN = 11; // LOW = enabled

const int ANALOG_PIN = A0;

static uint16_t grid[16][16];

void setMuxChannel(int s0, int s1, int s2, int s3, uint8_t ch) {
  digitalWrite(s0, (ch & 0x01) ? HIGH : LOW);
  digitalWrite(s1, (ch & 0x02) ? HIGH : LOW);
  digitalWrite(s2, (ch & 0x04) ? HIGH : LOW);
  digitalWrite(s3, (ch & 0x08) ? HIGH : LOW);
}

// Convenience wrappers
inline void selectRow(uint8_t r) { setMuxChannel(ROW_S0, ROW_S1, ROW_S2, ROW_S3, r); }
inline void selectCol(uint8_t c) { setMuxChannel(COL_S0, COL_S1, COL_S2, COL_S3, c); }

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  pinMode(ROW_S0, OUTPUT);
  pinMode(ROW_S1, OUTPUT);
  pinMode(ROW_S2, OUTPUT);
  pinMode(ROW_S3, OUTPUT);

  pinMode(COL_S0, OUTPUT);
  pinMode(COL_S1, OUTPUT);
  pinMode(COL_S2, OUTPUT);
  pinMode(COL_S3, OUTPUT);

  // If using EN pins:
  // pinMode(ROW_EN, OUTPUT);
  // pinMode(COL_EN, OUTPUT);
  // digitalWrite(ROW_EN, LOW); // enable
  // digitalWrite(COL_EN, LOW); // enable

  analogReadResolution(12); //12 bits, 0..4095 on Nano 33 BLE Sense
  //analogReference(AR_DEFAULT); // default 3.3V ref

  // Initialize zero selection
  selectRow(0);
  selectCol(0);

  Serial.println("Starting 16x16 pressure scan...");
}

// small wait after switching channels to let things settle
inline void settle() {
  // 100–300 microseconds is usually enough; tweak as needed
  delayMicroseconds(150);
}

void loop() {
  uint16_t maxVal = 0;
  uint8_t maxR = 0, maxC = 0;

  for (uint8_t r = 0; r < 16; r++) {

    selectRow(r);
    settle();

    for (uint8_t c = 0; c < 16; c++) {
       selectCol(c);
       settle();

    //   // Take a couple of samples and average for stability
      uint32_t acc = 0;

     // Serial.print("row ");
     // Serial.print(c);
     
  
      const int samples = 3;
    //  Serial.print(" samples: ");

      for (int k = 0; k < samples; k++) {
        uint32_t temp = analogRead(ANALOG_PIN);
        acc += temp;

     //   Serial.print(temp);
     //   Serial.print(",");

      }


      uint16_t val = acc / samples;
      
   //   Serial.print(" avg:");
    //  Serial.println(val);


       grid[r][c] = val;

      if (val > maxVal) {
        maxVal = val;
        maxR = r;
        maxC = c;
      }
     }
  }

  // Print summary “hot spot”
  Serial.print("Max at (row, col)=("); //,col
   Serial.print(maxR);
   Serial.print(",");
   Serial.print(maxC);
   Serial.print(")  value=");
   Serial.println(maxVal);

  // Print the 16x16 grid as CSV rows (one row per line)
  char buf[6]; // enough for "0000," + null terminator
  for (uint8_t r = 0; r < 16; r++) {
    for (uint8_t c = 0; c < 16; c++) {
      sprintf(buf, "%04d", grid[r][c]);
      Serial.print(buf);
      if (c < 15) Serial.print(',');
    }
    Serial.println();
  }
  Serial.println("---");

  delay(2000); // ~50 FPS; adjust to taste
}
