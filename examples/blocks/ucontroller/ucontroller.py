# coding: utf-8

"""
This example demonstrates the use of a UController Block for communicating with
a microcontroller over a USB connection It necessitates a microcontroller with
MicroPython installed to run, and does not require any specific Python module.
It was written for and tested on Adafruit's Huzzah ESP32 board, but should
work on other microcontrollers or boards with minor adjustments.

The UController Block can drive a microcontroller with MicroPython installed,
and running the microcontroller.py template adjusted to user's needs. It can
also drive a microcontroller running the Arduino microcontroller.ino script,
that can be found in Crappy's tool. The UController Block can send commands to
set values on the microcontroller, as well as read values acquired by the
microcontroller and send them to downstream Blocks.

Here, the UController Block drives an Adafruit Huzzah ESP32 microcontroller
running the microcontroller.py MicroPython script. This script makes the
integrated red light of the board blink, and keeps count of the number of
blinks. A Generator Block decides on the frequency of the blinking, and sends
increasing frequency values to the UController Block that sets them on the
microcontroller. The microcontroller also returns the blink count every 10
blinks to the UController Block, that then sends it to a Dashboard Block for
display.

Before starting this script, you first have to install MicroPython on your
microcontroller and upload the microcontroller.py script as the main.py file.
Depending on your microcontroller, OS and setup, you might have to adjust the
code below. Also, on Adafruit's Huzzah ESP32, the pin 13 is linked to an
integrated red LED. To see the effect of this example, you might have to change
the number of the driven pin in the microcontroller.py file and/or to connect a
LED to your microcontroller.

After starting the script, watch how the LED blinks at the target frequency,
and how the frequency increases every 10s. The blink count is also successfully
returned and displayed. This script automatically ends after 52s. You can also
hit CTRL+C to stop it earlier, but it is not a clean way to stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates the frequency command for the UController
  # Block. It outputs a constant signal whose value increases each 10s
  gen = crappy.blocks.Generator(
      # Generating constant signals of increasing values
      ({'type': 'Constant', 'condition': 'delay=10', 'value': 1},
       {'type': 'Constant', 'condition': 'delay=10', 'value': 2},
       {'type': 'Constant', 'condition': 'delay=10', 'value': 4},
       {'type': 'Constant', 'condition': 'delay=10', 'value': 8},
       {'type': 'Constant', 'condition': 'delay=10', 'value': 16}),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='freq',  # The label carrying the generated signal

      # Sticking to default for the other arguments
  )

  # This UController Block communicates with a microcontroller running the
  # microcontroller.py script. It receives the frequency command from the
  # Generator Block, and sets it as the blinking frequency for the GPIO on the
  # microcontroller. It also receives the blink count and the timestamp from
  # the microcontroller and sends it to the Dashboard Block for display
  micro = crappy.blocks.UController(
      labels=('nr',),  # The labels to read from the microcontroller, except
      # for the time label
      cmd_labels=('freq',),  # The command labels to transmit to the
      # microcontroller
      init_output={'nr': 0},  # The value to output for each label as long as
      # no value was received from the microcontroller
      t_device=True,  # The returned time is the one read from the
      # microcontroller, not the one of Crappy
      port='/dev/ttyUSB0',  # The port on which the microcontroller is
      # connected, might need to be adjusted depending on your OS, setup, and
      # microcontroller type
      baudrate=115200,  # This baudrate is fine for the ESP32, but may need to
      # be adjusted for other microcontrollers
      freq=30,  # Lowering the default frequency because it's just  demo

      # Sticking to default for the other arguments
  )

  # This Dashboard Block displays the timestamp and the number of blinks that
  # are transmitted by the UController Block
  dash = crappy.blocks.Dashboard(
      # The names of the labels to display
      ('t(s)', 'nr'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, micro)
  crappy.link(micro, dash)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
