// ============================
// Autonomous Swarm Robot - Arduino
// ============================

// Motor pins
int motor_left[] = {5, 6};
int motor_right[] = {7, 8};

// LED for status
int statusLED = 12;

// Ultrasonic Sensors
int trigFront = 9;
int echoFront = 10;
int trigLeft = 11;
int echoLeft = 3;
int trigRight = 4;
int echoRight = 2;

// Variables
long duration;
int distanceFront, distanceLeft, distanceRight;

// --------------------------- Setup
void setup() {
  Serial.begin(9600);

  // Motors
  for (int i = 0; i < 2; i++) {
    pinMode(motor_left[i], OUTPUT);
    pinMode(motor_right[i], OUTPUT);
  }

  // LED
  pinMode(statusLED, OUTPUT);
  digitalWrite(statusLED, LOW);

  // Ultrasonic
  pinMode(trigFront, OUTPUT);
  pinMode(echoFront, INPUT);
  pinMode(trigLeft, OUTPUT);
  pinMode(echoLeft, INPUT);
  pinMode(trigRight, OUTPUT);
  pinMode(echoRight, INPUT);
}

// --------------------------- Main Loop
void loop() {
  // 1️⃣ Read all sensors
  distanceFront = readUltrasonic(trigFront, echoFront);
  distanceLeft  = readUltrasonic(trigLeft, echoLeft);
  distanceRight = readUltrasonic(trigRight, echoRight);

  // 2️⃣ Send sensor data to AI (EDQ + SERAI simulation)
  Serial.print("FRONT:"); Serial.print(distanceFront);
  Serial.print(" LEFT:"); Serial.print(distanceLeft);
  Serial.print(" RIGHT:"); Serial.println(distanceRight);

  // 3️⃣ Read AI Decision from Serial (simulated for now)
  if (Serial.available() > 0) {
    char decision = Serial.read();

    if (decision == 'F') driveForward();
    else if (decision == 'L') turnLeft();
    else if (decision == 'R') turnRight();
    else if (decision == 'S') stopMotors();
  }

  delay(100); // Loop delay
}

// --------------------------- Functions

int readUltrasonic(int trigPin, int echoPin){
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  int distance = duration * 0.034 / 2;
  return distance;
}

void driveForward(){
  digitalWrite(motor_left[0], HIGH);
  digitalWrite(motor_left[1], LOW);
  digitalWrite(motor_right[0], HIGH);
  digitalWrite(motor_right[1], LOW);
  digitalWrite(statusLED, HIGH);
}

void turnLeft(){
  digitalWrite(motor_left[0], LOW);
  digitalWrite(motor_left[1], HIGH);
  digitalWrite(motor_right[0], HIGH);
  digitalWrite(motor_right[1], LOW);
  digitalWrite(statusLED, HIGH);
}

void turnRight(){
  digitalWrite(motor_left[0], HIGH);
  digitalWrite(motor_left[1], LOW);
  digitalWrite(motor_right[0], LOW);
  digitalWrite(motor_right[1], HIGH);
  digitalWrite(statusLED, HIGH);
}

void stopMotors(){
  for (int i = 0; i < 2; i++){
    digitalWrite(motor_left[i], LOW);
    digitalWrite(motor_right[i], LOW);
  }
  digitalWrite(statusLED, LOW);
}
