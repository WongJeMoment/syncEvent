#include <Servo.h>

Servo gm6020;
const int pwmPin = 9;

void setup() {
  gm6020.attach(pwmPin);
  delay(2000);

  gm6020.writeMicroseconds(1578 );  // 明显大于 1500，转速高，确保电机会响应
}

void loop() {
}
