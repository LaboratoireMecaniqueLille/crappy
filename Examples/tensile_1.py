# coding: utf-8

"""
Detailed example showing a simple tensile test.

This example uses a Labjack T7 board to send the position command to a tensile
machine. This was tested on an Instron 5882, but any tensile machine taking an
external command can be used.

Required hardware:
  - Tensile machine able to take an external command
  - DAQ board (here a Labjack T7)
"""

import crappy

if __name__ == "__main__":
  SPEED = 5  # mm/min
  FORCE_GAIN = 100  # N/V
  POS_GAIN = 5  # mm/V

  # The path we will use in the generator
  path = {
    "type": "ramp",
    "speed": SPEED / 60,  # Convert to mm/s (in ramps, speed is always
    # in unit/s)
    "condition": None}  # No exit condition: we will stop the test manually

  # Let's create the first block: the generator
  generator = crappy.blocks.Generator([path], cmd_label='cmd')

  # Now the DAQ board we will interact with. Let's define the channels first
  # First input: the force sensor
  force = {'name': 'AIN0', 'gain': FORCE_GAIN}
  # Second input: the position sensor
  # make_zero asks the board to read the value at the beginning and subtract
  # if from the measured value to make it relative to the beginning of the test
  pos = {'name': 'AIN1', 'gain': POS_GAIN, 'make_zero': True}
  # Only output: the command sent to the machine
  # Because TDACx is necessarily an output and AINx an input, the block will
  # guess the direction of each channel. If ambiguous it must be specified with
  # direction=...  (False for input, True for output)
  cmd = {'name': 'TDAC0', 'gain': POS_GAIN}
  # Now let us create the block: the first arg is the name of the InOut object
  # to use. The block will instantiate it itself ! No need to create it here
  # Then we give the channels, the InOut will automatically detect that it has
  # two inputs and one output
  # The output labels always start with the time in second so we need 3 outputs
  daq = crappy.blocks.IOBlock('Labjack_t7', channels=[force, pos, cmd],
                              labels=['t(s)', 'F(N)', 'Position(mm)'],
                              cmd_labels=['cmd'])
  # Now we link them: the output of the board will be read from the label 'cmd'
  # Note that links are one-way only, so the order is important
  crappy.link(generator, daq)

  # Now we would like to save the measurements from the machine
  rec = crappy.blocks.Recorder('results.csv')
  crappy.link(daq, rec)

  # And why not display the force in real time during the test
  graph = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  crappy.link(daq, graph)

  # Now that all the blocks are created and linked, the test can be started
  # Do not forget to enable the external command on the machine beforehand
  crappy.start()
  # To interrupt the test, simply press ctrl+C in the terminal
  # It will stop all the blocks and end the program
