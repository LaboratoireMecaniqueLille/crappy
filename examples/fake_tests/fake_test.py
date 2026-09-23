# coding: utf-8

"""
This example demonstrates how to use a FakeMachine Block to emulate a fake
tensile test. It is very similar to the blocks/fake_machine.py example. It does
not make use of video-processing Blocks, unlike the other examples in the same
folder.

It requires matplotlib to run.

The test ends automatically after the loading path completes. Click the stop
button to end it earlier.
"""

import crappy

if __name__ == "__main__":

  # This Generator Block generates the speed command to send to the FakeMachine
  # Block. The signal makes the FakeMachine stretch the fake sample in cycles
  # of increasing amplitude
  gen = crappy.blocks.Generator(
      # Generating pairs of constant paths of opposite value, with increasing
      # amplitudes
      path=sum([[{'type': 'Constant', 'value': 5 / 60,
                  'condition': f'Exx(%)>{i / 3}'},
                 {'type': 'Constant', 'value': -5 / 60,
                  'condition': 'F(N)<0'}]
                for i in range(1, 6)], list()),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='cmd',  # The label carrying the generated signal

      # Sticking to default for the other arguments
  )

  # This FakeMachine Block takes the speed command from the Generator Block
  # as input and outputs the strain and force to the Grapher Block
  machine = crappy.blocks.FakeMachine(
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block plots the force-strain data it receives from the
  # FakeMachine Block. It can do so because both channels are on the same time
  # basis
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot in the grapher window
      ('Exx(%)', 'F(N)'),

      # Sticking to defaults for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, machine)
  crappy.link(machine, gen)
  crappy.link(machine, graph)

  # This Block provides a clean way to stop the test before it ends
  stop = crappy.blocks.StopButton()

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
