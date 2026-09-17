# coding: utf-8

"""
This example demonstrates the use of the Grapher Block. It does not require any
specific hardware to run, but necessitates the matplotlib Python module to be
installed.

The Grapher Block displays the data it receives from one or more Blocks in a
scatter plot. It can display the values of a label against time or another
label if both labels are received from the same Link.

Here, two Graphers are instantiated. The first displays data against time, the
second displays a force-strain curve with force on the y-axis and strain on the
x-axis. To showcase several Grapher options, the first limits the number of
data points it can display, and the second displays non-interpolated points.

After starting this script, just watch the curves evolve on the two Graphers.
This demo ends after 27 seconds. Click the stop button to end the demo early.
"""

import crappy

if __name__ == '__main__':

  # This Generator generates a signal for driving the FakeMachine
  # It simply outputs a constant speed
  gen = crappy.blocks.Generator(
      # Generating the constant signal with a value of 0.1 during 25 seconds
      ({'type': 'Constant',
        'value': 0.1,
        'condition': 'delay=25'},),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='cmd',  # The label carrying the command value
      spam=True,  # Sending a value at each loop, to obtain nice graphs

      # Sticking to default for the other arguments
  )

  # This Block emulates the behavior of a tensile test machine
  # It is used here because it generates data that can be plotted in a
  # force-strain curve
  machine = crappy.blocks.FakeMachine(
      mode='speed',  # Driving the fake machine in speed, not in position
      cmd_label='cmd',  # The label carrying the speed command
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher displays the command speed from the Generator, as well as the
  # position data from the FakeMachine
  # It demonstrates the possibility of displaying simultaneously several
  # curves, moreover with data coming from two Blocks
  graph_1 = crappy.blocks.Grapher(
      # Listing all the labels to display
      ('t(s)', 'cmd'), ('t(s)', 'x(mm)'),
      length=150,  # Limiting the display to the last 150 chunks of data

      # Sticking to default for the other arguments
  )

  # This Grapher displays on the same graph the force and the strain it
  # receives from the FakeMachine
  # It demonstrates the possibility to put any label on the x and y axes, as
  # long as they are received from the same Link
  graph_2 = crappy.blocks.Grapher(
      # Listing all the labels to display
      ('Exx(%)', 'F(N)'),
      interp=False,  # Displaying the data points and not interpolating them

      # Sticking to default for the other arguments
  )

  # Linking the Blocks together so that each one sends and receives the correct
  # information
  crappy.link(gen, machine)
  crappy.link(gen, graph_1)
  crappy.link(machine, graph_1)
  crappy.link(machine, graph_2)

  # This Block provides a clean way to stop the test before it ends
  stop = crappy.blocks.StopButton()

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
