float x, y, z, Vx, Vy, Vz;
float Vcc = 3.3;
int numReadings = 100*30;
int readingCounter = 0;
float M[] = {0, 0, 0};
float N[] = {0, 0, 0};
float newValue = 0;

void setup() {
  Serial.begin(9600);
  dacWrite(25,0);
}


float midlingsFilter(float x, float * arr){
  newValue = arr[0];
  newValue += arr[1];
  newValue += arr[2];
  newValue += x;
  arr[0] = arr[1];
  arr[1] = arr[2];
  arr[2] = x;
  return newValue/4;
}


void loop() {
  if (readingCounter < numReadings){
  x = analogRead(ADC1);
  y = analogRead(ADC2);
  //z = analogRead(ADC2);
  Vx = (x * Vcc/4095) - 1.65;
  Vy = (y * Vcc/4095) - 1.65;
  //Vz = (z * Vcc/4095) + 0.4 - 1.65;
  Serial.print((Vx/0.8 + 0.15), 4);
  Serial.print(" ");
  Serial.print((Vy/0.8 + 0.05), 4);
  //Serial.print((Vz/0.8 + 0.1), 4);
  Serial.print(" ");
  Serial.print(-1);
  Serial.print(" ");
  Serial.println(1);
  delay(10); // fs = 100
  readingCounter++;
 }
}
