/* To use this template, replace the lines between the >>> and <<< by your own
code. See the examples/blocks/ucontroller folder for a running example.
Here, this example simply sets a GPIO low and high at a given frequency and
sends back the number of cycles to the PC every 10 cycles. Connect a LED to the
GPIO to see the result ! */


/* Reset the microcontroller, so that it is then stuck again in the first
infinite loop and waiting for the 'go' message. */
void(* resetFunc) (void) = 0;


// Structure holding the attributes of the labels and commands
struct Item {
   int id;
   float value;
   String name;
};

// If true, the timestamp is returned to the PC along with the labels
bool send_t;


// Converts float to 4 bytes in little endian, independently from the
// microcontroller endianness
byte* float_to_bytearray(float f) {
    byte* ret = (byte *) malloc(4 * sizeof(byte));
    unsigned int asInt = *((int *) &f);

    for (int i = 0; i < 4; i++) {
        ret[i] = (asInt >> 8 * i) & 0xFF;
    }

    return ret;
}


// Arrays containing the commands and labels
struct Item Labels[10];
struct Item Commands[10];


/* All the following functions either modify or return attributes of the
commands and labels */

void set_label_value (float value, String label) {
  size_t len = sizeof(Labels)/sizeof(Labels[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Labels[i].name == label)
    {
      Labels[i].value = value;
      break;
    }
  }
}


float get_label_value (String label) {
  size_t len = sizeof(Labels)/sizeof(Labels[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Labels[i].name == label)
    {
      return Labels[i].value;
    }
  }
}


int get_label_id (String label) {
  size_t len = sizeof(Labels)/sizeof(Labels[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Labels[i].name == label)
    {
      return Labels[i].id;
    }
  }
}


void set_cmd_value (float value, String cmd) {
  size_t len = sizeof(Commands)/sizeof(Commands[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Commands[i].name == cmd)
    {
      Commands[i].value = value;
      break;
    }
  }
}


void set_cmd_value_by_id (float value, int id) {
  size_t len = sizeof(Commands)/sizeof(Commands[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Commands[i].id == id)
    {
      Commands[i].value = value;
      break;
    }
  }
}


float get_cmd_value (String cmd) {
  size_t len = sizeof(Commands)/sizeof(Commands[0]);
  for (int i = 0; i < 10; ++i)
  {
    if (Commands[i].name == cmd)
    {
      return Commands[i].value;
    }
  }
}


/* Sends back data to the PC.

If send_t is true, also sends back the timestamp in milliseconds. The time
is encoded as an integer, the data as a float. Inbetween, a signed char
indicates the index of the label that is being sent. This index has been set
by the PC and sent in the labels dict.

See below for a use case. */

void send_to_pc(float value, String label) {
  if (send_t)
  // Sending timestamp and data
  {
    int timestamp = (int) millis();
    // Converting float to byte array
    byte* ptr = float_to_bytearray(value);
    byte value_to_send[4] = {ptr[0], ptr[1], ptr[2], ptr[3]};
    // Converting int to byte array
    byte time_to_send[4] = {
      timestamp & 0xFF,
      (timestamp >> 8) & 0xFF,
      (timestamp >> 16) & 0xFF,
      (timestamp >> 24) & 0xFF};
    Serial.write(time_to_send, 4);
    Serial.write((signed char) get_label_id(label));
    Serial.write(value_to_send, 4);
    // Freeing memory allocated by float_to_bytearray
    free(ptr);
  }
  else
  // Sending only data
  {
    // Converting float to byte array
    byte* ptr = float_to_bytearray(value);
    byte value_to_send[4] = {ptr[0], ptr[1], ptr[2], ptr[3]};
    Serial.write((signed char) get_label_id(label));
    Serial.write(value_to_send, 4);
    // Freeing memory allocated by float_to_bytearray
    free(ptr);
  }
}

/* Enters an infinite loop and exits only upon reception of 'goXY'. X is the
number of command labels, Y is the number of labels. The labels are then
received and stored. This setup prevents the program from doing anything before
it is told to. */

void setup()
{
  Serial.begin (115200);
  send_t = false;
  int nb_cmd;
  int nb_lab;

  while (1)
  {
    // Program has been launched on PC
    if (Serial.available())
    {
      String msg = Serial.readStringUntil('\r\n');
      msg.replace("\r", "");
      if (msg.startsWith("go"))
      {
        nb_cmd = String(msg.charAt(2)).toInt();
        nb_lab = String(msg.charAt(3)).toInt();
        break;
      }
    }
  }

  // Getting the commands
  for (int i = 0; i < nb_cmd; ++i)
  {
    String msg = Serial.readStringUntil('\r\n');
    msg.replace("\r", "");
    int id = String(msg.charAt(0)).toInt();
    String cmd = msg.substring(1);

    Commands[i].id = id;
    Commands[i].name = cmd;

    delay(50);
  }

  // Getting the labels
  for (int i = 0; i < nb_lab; ++i)
  {
    String msg = Serial.readStringUntil('\r\n');
    msg.replace("\r", "");
    int id = String(msg.charAt(0)).toInt();
    String label = msg.substring(1);

    Labels[i].id = id;
    Labels[i].name = label;

    // If "t(s)"" in labels, the program should send back timestamps
    if (label == String("t(s)"))
    {
      send_t = true;
    }

    delay(50);
  }

  // >>>>>>>>

  // Here we initialize the variables we want to use
  pinMode (13, OUTPUT);

  set_cmd_value(1, String("freq"));
  set_label_value(0, String("nr"));

  // <<<<<<<<

}


/* The main loop of this script.
It reads the incoming messages, and does a user-defined action when no message
is waiting. */

void loop()
{
  // Reads the incoming messages if any is waiting
  if (Serial.available())
  {
    String msg = Serial.readStringUntil('\r\n');
    msg.replace("\r", "");

    // Upon reception of 'stop!', the program ends
    if (msg == String("stop!"))
    {
      resetFunc();
    }


    /* Acquiring commands
    Their names are those of the cmd_labels in the UController block
    In the example only freq can be updated as it is the only element of
    cmd_labels */
    int id = String(msg.charAt(0)).toInt();
    float value = msg.substring(1).toFloat();
    set_cmd_value_by_id(value, id);

  }

  while (!Serial.available())
  {
    /* Here should be the main task of your script
    It will be called repeatedly, but the loop will be interrupted upon
    reception of a command */

    // >>>>>>>>

    // Blinking the GPIO
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500 / get_cmd_value(String("freq")));
    digitalWrite(LED_BUILTIN, LOW);
    delay(500 / get_cmd_value(String("freq")));

    /* Sending back the number of cycles (count) under the label 'nr'
    Only the labels present in the labels argument of the UController block
    can be sent back
    Make sure that your Crappy and MicroPython scripts use the same naming */
    if ((int) get_label_value(String("nr")) % 10 == 0)
    {
      /* Example of a call to send_to_pc
      First argument is the data to send, second is its label
      Also sends back the associated timestamp if send_t is true */
      send_to_pc(get_label_value(String("nr")), String("nr"));
    }

    set_label_value(get_label_value(String("nr")) + 1, String("nr"));

    // <<<<<<<<
  }
}
