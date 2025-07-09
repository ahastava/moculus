int led = 13;
   
void setup() {
  // put your setup code here, to run once:
  pinMode(led, OUTPUT);
    Serial.begin(9600); 
}

void loop() {

  // put your main code here, to run repeatedly:
  digitalWrite(led, HIGH);
  Serial.print("HIGH: ");
  Serial.print(HIGH);
  delay(25);
  digitalWrite(led, LOW);
  Serial.print("LOW: ");
  Serial.print(LOW);
  delay(1000);
  
}
