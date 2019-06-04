#define NTC_R_10000
#include "ntc.h"

float sensorValue, Vntc, Rntc, temp;
float oldValue, newValue = 0;
int numReadings = 100*10;
int readingCounter = 0;
float Vcc = 3.3;
float M[] = {0, 0, 0};


void setup() {
  Serial.begin(9600);
  dacWrite(25,0);
}


float lowpassFilter(float x){
  oldValue = newValue;
  newValue = (2.452372752527856026e-1 * x)
        + (0.50952544949442879485 * oldValue);
  return (oldValue + newValue); // Cutoff 10 Hz
}


float midlingsFilter(float x){
  newValue = M[0];
  newValue += M[1];
  newValue += M[2];
  newValue += x;
  M[0] = M[1];
  M[1] = M[2];
  M[2] = x;
  return newValue/4;
}


void loop() {
  if (readingCounter < numReadings){
  sensorValue = analogRead(ADC1);
  Vntc = sensorValue * Vcc/4095;
  Rntc = NTC_REF_R * (Vntc/(Vcc - Vntc));
  temp = ntcToTemp(Rntc) - 273.15;
  Serial.print((temp), 4);
  Serial.print(" ");
  Serial.print(0);
  Serial.print(" ");
  Serial.println(40);
  delay(10); // fs = 100 Hz
  readingCounter++;
 }
}
