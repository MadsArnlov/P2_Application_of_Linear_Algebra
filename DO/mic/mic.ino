float sensorValue, Vmic;
float oldValueL, oldValueH, newValueL, newValueH = 0;
int numReadings = 1000*5; // 10 seconds recording time
int readingCounter = 0;
int scale = 10;

void setup() {
  Serial.begin(115200);
  dacWrite(25,0);
}


float lowpassFilter(float x){ // Cutoff 420 Hz
  oldValueL = newValueL;
  newValueL = (7.956991756997355836e-1 * x)
         + (-0.59139835139947094511 * newValueL);
  return (oldValueL + newValueL);
}


float highpassFilter(float x){ // Cutoff 80 Hz
  oldValueH = newValueH;
  newValueH = (7.956991756997355836e-1 * x);
        + (0.59139835139947116716 * oldValueH);
  return (newValueH - oldValueH);
}


void loop() {
  if (readingCounter < numReadings){
  sensorValue = analogRead(ADC1);
  Vmic = sensorValue * 5/4095; // amplifying the signal by 10
  Vmic = highpassFilter(Vmic);
  Vmic = lowpassFilter(Vmic);
  Serial.println(Vmic*scale, 4); // Passband: 80 Hz to 420 Hz
  delay(1); // fs = 1000 Hz
  readingCounter++;
 }
}
