//Define ultrasonic sensor pins
#define trigPin 1
#define echoPin 2

//Define motor control Pins
#define enA 10
#define in1 8
#define in2 9

#define enB 5
#define in3 6
#define in4 7

// defines variables for ulstrasonic sensor
long duration;
int distance;

//value sent over via bluetooth
char lastValue;
char blueToothVal;

int l_flag = 0;
int r_flag = 0;

char l_val;
char r_val;

  int linear;
  int angular;

  
//===================================================

void setup()
{
  Serial.begin (9600);
  pinMode(LED_BUILTIN, OUTPUT);

  
  // set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  //Set Ultrasonic Sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  
}

//==================================================

void dist()
{
  // Clears the trigPin
digitalWrite(trigPin, LOW);
delayMicroseconds(2);
// Sets the trigPin on HIGH state for 10 micro seconds
digitalWrite(trigPin, HIGH);
delayMicroseconds(10);
digitalWrite(trigPin, LOW);
// Reads the echoPin, returns the sound wave travel time in microseconds
duration = pulseIn(echoPin, HIGH);
// Calculating the distance
distance= duration*0.034/2;
}

//========================================

void blue_tooth_receive()
{
  int buf_size = Serial.available();

  if(buf_size > 0)
  {//if there is data being recieved
    blueToothVal=Serial.read(); //read it
    Serial.println(blueToothVal);
  if (r_flag) {
    r_val = blueToothVal;
    r_flag = 0;
  }
  else if (l_flag) {
    l_val = blueToothVal;
    r_flag = 1;
    l_flag = 0; 
  }
  else if (blueToothVal == 'H') {
    l_flag = 1;
  }
  
  }

  //delay(30);
}


//=======================================

void motor_control()
{

  
  if (l_val == 'C')
  {
        linear = 1;
  }
  else if (l_val == 'A')
  {
        linear = -1;      
  }
    
  else if (l_val == 'B')
  {
        linear = 0;
  }

//====================================
  
  if (r_val == 'C')
  {     
        angular = 1;

  }
  else if (r_val == 'A')
  {
        angular = -1;
        
  }
  else if (r_val == 'B')
  {
        angular = 0;

  }
//=====================================

int cmd_left = linear - 2*angular;
int cmd_right = linear + 2*angular;

  if (cmd_left > 0)
  {
   digitalWrite(in1, HIGH);
   digitalWrite(in2, LOW);
  }
  else if (cmd_left < 0)
  {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
  else if (cmd_left == 0)
  {
    digitalWrite(in1, LOW);
     digitalWrite(in2, LOW);
  }

  if (cmd_right > 0)
  {
   digitalWrite(in3, HIGH);
   digitalWrite(in4, LOW);
  }
  else if (cmd_left < 0)
  {
     digitalWrite(in3, LOW);
     digitalWrite(in4, HIGH);
  }
  else if (cmd_left == 0)
  {
    digitalWrite(in3, LOW);
     digitalWrite(in4, LOW);
  }

  analogWrite(enA, 125*abs(cmd_left));
  analogWrite(enB, 125*abs(cmd_right));

/*  
  Serial.print("linear: ");
  Serial.println(linear);
  Serial.print("Angular: ");
  Serial.println(angular);



  Serial.print("Commanded Left: ");
  Serial.println(cmd_left);
  Serial.print("Commanded Right: ");
  Serial.println(cmd_right);
  */

}

//======================================

void loop()
{

  blue_tooth_receive();
  motor_control();

} 
