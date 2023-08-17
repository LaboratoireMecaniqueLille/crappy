# coding: utf-8

"""
This example is an extension of the fake_test.py script, that includes
digital image correlation to emulate the extension measurement on the fake
sample. Its goal is to emulate a fake tensile test driven with Crappy and
featuring digital image correlation.

It requires matplotlib, opencv-python, scikit-image and Pillow to run.
"""

import crappy


def plastic_law(_: float) -> float:
  """No plastic law in this simple example."""

  return 0.


if __name__ == "__main__":

  # Loading the example image for performing image correlation
  # This image is distributed with Crappy
  img = crappy.resources.speckle

  # This Generator Block generates the speed command to send to the FakeMachine
  # Block. The signal is so that the FakeMachine will stretch the fake sample
  # in cycles of increasing amplitude
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
  # as an input, and outputs the extension and the stress to the DISCorrel
  # Block
  machine = crappy.blocks.FakeMachine(
      rigidity=5000,  # The stiffness of the fake sample
      l0=20,  # The initial length of the fake sample
      max_strain=17,  # The fake sample breaks passed this strain value
      sigma={'F(N)': 0.5},  # Adding noise to the effort signal
      plastic_law=plastic_law,  # Adding the custom plastic law to the model of
      # the fake sample
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This DISCorrel Block computes the extension using image correlation, on an
  # image stretched based on the extension values received from the FakeMachine
  # Block. The pair FakeMachine + DISCorrel Blocks thus models the behavior of
  # an equivalent tensile test setup
  dis = crappy.blocks.DISCorrel(
      '',  # The name of Camera to open is ignored because image_generator is
      # given
      display_images=True,  # The displayer window will allow to follow the
      # spots on the acquired images
      labels=['t(s)', 'meta', 'x', 'y', 'meas_Exx(%)', 'meas_Eyy(%)'],
      image_generator=crappy.tool.ApplyStrainToImage(img),  # This argument
      # makes the Block generate fake strain on the given image, only useful
      # for demos
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block plots the extension data it receives from the
  # DISCorrel Block
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'meas_Exx(%)'), ('t(s)', 'meas_Eyy(%)'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, machine)
  crappy.link(machine, dis)
  crappy.link(machine, gen)
  crappy.link(dis, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
