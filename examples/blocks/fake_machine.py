# coding: utf-8

"""
This example demonstrates the use of the FakeMachine Block. It does not require
any specific hardware to run, but necessitates the matplotlib Python module to
be installed.

This Block simulates the behavior of a tensile test machine. It takes a speed
or position command as an input, and outputs the position, force and strain on
the simulated sample. The parameters of the samples can be adjusted.

In this example, a monotonic cyclic stretching with relaxation to 0 force is
applied to the FakeMachine bu a Generator Block. The force-strain curve of
the test is displayed by a Grapher Block.

After starting this script, watch the force-strain curve being displayed. After
a few cycles, the plastic behavior of the sample should be clearly visible.
This demo ends after a few minutes, or it can be stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator generates the command for driving the FakeMachine Block
  # It emulates a cyclic monotonic stretching with increasing strain levels
  # and relaxation until 0 force is reached
  gen = crappy.blocks.Generator(
      # To build the path, using list addition and f-string formatting
      # The first path drives the FakeMachine to the desired strain ,the second
      # releases it to a 0 effort state
      sum([[{'type': 'Constant',
             'value': 5 / 60,
             'condition': f'Exx(%)>{i / 3}'},
            {'type': 'Constant',
             'value': -5 / 60, 'condition': 'F(N)<0'}] for i in range(1, 6)],
          []),
      spam=False,  # Only sending a new command when it differs from the
      # previous
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='cmd',  # The label carrying the generated command

      # Sticking to default for the other arguments
      )

  # This FakeMachine Block emulates the behavior of a tensile test setup
  # It takes a speed command as input, and outputs the force, position and
  # strain on the emulated sample
  machine = crappy.blocks.FakeMachine(
      mode='speed',  # Driving the fake machine in speed mode
      cmd_label='cmd',  # The label carrying the speed command
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # This Grapher plots the stress-strain (actually force-strain) curve of the
  # sample emulated by the FakeMachine
  graph = crappy.blocks.Grapher(('Exx(%)', 'F(N)'))

  # Linking the Blocks together so that each one sends and received the correct
  # information
  # The Generator drives the FakeMachine, but also takes decision based on its
  # feedback
  crappy.link(gen, machine)
  crappy.link(machine, gen)
  crappy.link(machine, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
