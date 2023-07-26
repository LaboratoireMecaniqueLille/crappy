# coding: utf-8

"""
This example demonstrates the use of the Grapher Block. It does not require any
specific hardware to run, but necessitates the matplotlib Python module to be
installed.

The Grapher Block displays the data it received from 1 or more Blocks in a
scatter plot. It can display the values of a label vs time, or vs another label
if both labels are received from the same Link.

Here, two Graphers are instantiated. The first one displays data vs time, the
second one displays a stress-strain curve with the stress label on the y-axis
and the strain label on the x-axis. Just for showcasing some options of the
Grapher, the first one limits the number of datapoints it can display, and the
second displays non-interpolated data points.

After starting this script, just watch the curves evolve on the two Graphers.
This demo ends after 27s, or it can be stopped before by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator generates a signal for driving the FakeMachine
  # It simply outputs a constant speed
  gen = crappy.blocks.Generator(
      # Generating the constant signal with a value of 0.1 during 25s
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
  # stress-strain curve
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

  # This Grapher displays on the same graph the stress and the strain it
  # receives from the FakeMachine
  # It demonstrates the possibility to put any label on the x and y axes, as
  # long as they are received from the same Link
  graph_2 = crappy.blocks.Grapher(
      # Listing all the labels to display
      ('Exx(%)', 'F(N)'),
      interp=False,  # Displaying the data points and not interpolating them

      # Sticking to default for the other arguments
  )

  # Linking the Blocks together so that each one sends and received the correct
  # information
  crappy.link(gen, machine)
  crappy.link(gen, graph_1)
  crappy.link(machine, graph_1)
  crappy.link(machine, graph_2)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
