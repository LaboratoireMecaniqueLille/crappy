# coding: utf-8

from machine import reset
from select import poll, POLLIN
from sys import stdin, stdout
from utime import sleep_ms, ticks_ms
from struct import pack

"""To use this template, replace the lines between the >>> and <<< by your own 
code. See the examples/blocks/ucontroller folder for a running example.
Here, this example simply sets a GPIO low and high at a given frequency and 
sends back the number of cycles to the PC every 10 cycles. Connect a LED to the
GPIO to see the result !
"""

# Creating a poll object
p = poll()
p.register(stdin, POLLIN)
msg = ''


def read():
  """Method for reading the labels coming from the PC at the beginning of the
  program."""

  i = 0
  while True:
    val = stdin.readline().strip()
    if val:
      return val
    sleep_ms(50)
    if i > 9:
      break
    i += 1


"""Enters an infinite loop and exits only upon reception of 'goXY'. X is the
number of command labels, Y is the number of labels. The labels are then 
received and stored. This setup prevents the program from doing anything before 
it is told to."""
while True:
  if p.poll(0):
    msg = stdin.readline().strip()
    # Program has been launched on PC
    if msg.startswith('go'):
      commands = {}
      # Getting the commands
      for _ in range(int(msg[2])):
        value = read()
        commands[int(value[0])] = str(value[1:])
      labels = {}
      # Getting the labels
      for _ in range(int(msg[3])):
        value = read()
        labels[str(value[1:])] = int(value[0])
      # If 't(s)' in labels, the program should send back timestamps
      send_t = 't(s)' in labels
      msg = ''
      break
  sleep_ms(50)

# >>>>>>>>

# Here we initialize the variables we want to use
from machine import Pin
freq = 1
count = 0
pin = Pin(13, Pin.OUT)
pin.off()

# <<<<<<<<


def send_to_pc(var, label):
  """Sends back data to the PC.

  If send_t is True, also sends back the timestamp in milliseconds. The time
  is encoded as an integer, the data as a float. Inbetween, a signed char
  indicates the index of the label that is being sent. This index has been set
  by the PC and sent in the labels dict.

  See below for a use case.
  """

  try:
    if send_t:
      stdout.buffer.write(pack('<ibf', ticks_ms(), labels[label], var))
    else:
      stdout.buffer.write(pack('<bf', labels[label], var))
  except KeyError:
    pass


"""The main loop of this script.
It reads the incoming messages, and does a user-defined action when no message
is waiting."""
while True:
  # Reads the incoming messages if any is waiting
  if p.poll(0):
    msg = stdin.readline().strip()

  # Upon reception of 'stop!', the program ends
  if msg == 'stop!':
    break

  # Acquiring commands
  # Their names are those of the cmd_labels in the UController block
  # In the example only freq can be updated as it is the only element of
  # cmd_labels
  if msg:
    globals()[commands[int(msg[0])]] = float(msg[1:])
    msg = ''

  while not p.poll(0):
    # Here should be the main task of your script
    # It will be called repeatedly, but the loop will be interrupted upon
    # reception of a command

    # >>>>>>>>

    # Blinking the GPIO
    pin.on()
    sleep_ms(int(500 / freq))
    pin.off()
    sleep_ms(int(500 / freq))

    # Sending back the number of cycles (count) under the label 'nr'
    # Only the labels present in the labels argument of the UController block
    # can be sent back
    # Make sure that your Crappy and MicroPython scripts use the same naming
    if not count % 10:
      # Example of a call to send_to_pc
      # First argument is the data to send, second is its label
      # Also sends back the associated timestamp if send_t is True
      send_to_pc(count, 'nr')

    count += 1

    # <<<<<<<<

"""Reset the microcontroller, so that it is then stuck again in the first 
infinite loop and waiting for the 'go' message."""
reset()
