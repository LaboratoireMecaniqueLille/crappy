# coding: utf-8

"""
This example extends the fake_test.py script with video extensometry to emulate
strain measurements on the fake sample. Its goal is to emulate a tensile
test driven with Crappy and featuring video extensometry.

It requires matplotlib, opencv-python, scikit-image and Pillow to run.

The test ends automatically after the loading path completes. Click the stop
button to end it earlier.
"""

import crappy


def plastic_law(_: float) -> float:
  """No plastic law in this simple example."""

  return 0.


if __name__ == "__main__":

  # Loading the example image for performing video-extensometry
  # This image is distributed with Crappy
  img = crappy.resources.ve_markers

  # This Generator Block generates the speed command to send to the FakeMachine
  # Block. The signal makes the FakeMachine stretch the fake sample in cycles
  # of increasing amplitude
  gen = crappy.blocks.Generator(
      # Generating pairs of constant paths of opposite value, with increasing
      # amplitudes
      path=sum([[{'type': 'Constant', 'value': 5 / 60,
                  'condition': f'Exx(%)>{5 * i}'},
                 {'type': 'Constant', 'value': -5 / 60, 'condition': 'F(N)<0'}]
                for i in range(1, 5)], list()),
      cmd_label='cmd',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This FakeMachine Block takes the speed command from the Generator Block
  # as input and outputs the strain and force to the VideoExtenso
  # Block
  machine = crappy.blocks.FakeMachine(
      rigidity=5000,  # The stiffness of the fake sample
      l0=20,  # The initial length of the fake sample
      max_strain=17,  # The fake sample breaks past this strain value
      sigma={'F(N)': 0.5},  # Adding noise to the force signal
      plastic_law=plastic_law,  # Adding the custom plastic law to the model of
      # the fake sample
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This VideoExtenso Block computes strain using video extensometry on an
  # image stretched according to values from FakeMachine. Together, the
  # FakeMachine and VideoExtenso Blocks model a tensile test setup
  ve = crappy.blocks.VideoExtenso(
      '',  # The name of Camera to open is ignored because image_generator is
      # given
      display_images=True,  # The displayer window follows the
      # spots on the acquired images
      blur=False,  # No blurring in this simple example
      image_generator=crappy.tool.ApplyStrainToImage(img),  # This argument
      # makes the Block generate fake strain on the given image, only useful
      # for demos
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block plots the strain data it receives from the
  # VideoExtenso Block
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, machine)
  crappy.link(machine, ve)
  crappy.link(machine, gen)
  crappy.link(ve, graph)

  # This Block provides a clean way to stop the test before it ends
  stop = crappy.blocks.StopButton()

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
