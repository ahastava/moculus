

void setup() {

String stringOne = "Hello String";    


  Serial.begin(115200);
  delay(1000);
 Serial.println("asdf");
  Serial.println(stringOne);


}



void loop() {

String stringOne = "Hello String";   
  delay(1000);  // Give time for serial to initialize
  Serial.println("asdf11");
  Serial.println(stringOne);


}