# coding: utf-8

"""
Demonstration of a DIC controlled test.

This program is intended as a demonstration and is fully virtual. The strain is
measured using DIC and a PID controller drives the machine to apply a ramp of
strain.

No hardware required.
Requires the cv2 module to be installed.
"""

import crappy
import numpy as np
import cv2


class Apply_strain_img:
  """This class reshapes an image depending on input strain values."""

  def __init__(self,
               image: np.ndarray) -> None:
    """Sets the base image and initializes the necessary objects to use."""

    self._img = image

    # Building the lookup arrays for the cv2.remap method
    height, width, *_ = image.shape
    orig_x, orig_y = np.meshgrid(range(width), range(height))
    # These arrays correspond to the original state of the image
    self._orig_x = orig_x.astype(np.float32)
    self._orig_y = orig_y.astype(np.float32)

    # These arrays are meant to be added to the original image ones
    # If added as is, they correspond to a 100% strain state in both directions
    self._x_strain = self._orig_x * width / (width - 1) - width / 2
    self._y_strain = self._orig_y * height / (height - 1) - height / 2

  def __call__(self, exx: float, eyy: float) -> np.ndarray:
    """Returns the reshaped image, based on the given strain values."""

    exx /= 100
    eyy /= 100

    # The final lookup table is the sum of the original state ones plus the
    # 100% strain one weighted by a ratio
    transform_x = self._orig_x - (exx / (1 + exx)) * self._x_strain
    transform_y = self._orig_y - (eyy / (1 + eyy)) * self._y_strain

    return cv2.remap(self._img, transform_x, transform_y, 1)


def plastic_law(_: float) -> float:
  """No elastic law in this simple example."""

  return 0.


if __name__ == "__main__":
  img = crappy.resources.speckle

  speed = .05  # Strain rate (%/s)

  # Simply create a sawtooth signal at 0.05 units/s between 0 and 1
  # Note that this generator takes no feedback
  generator = crappy.blocks.Generator(path=[
    {'type': 'CyclicRamp', 'speed1': speed, 'condition1': 'target_Exx(%)>1',
     'speed2': -speed, 'condition2': 'target_Exx(%)<0', 'cycles': 3,
     'init_value': 0}], cmd_label='target_Exx(%)')
  crappy.link(generator, generator)

  # The Block emulating the behavior of a tensile test machine
  machine = crappy.blocks.FakeMachine(rigidity=5000, l0=20, max_strain=1.7,
                                      sigma={'F(N)': 0.5},
                                      plastic_law=plastic_law, cmd_label='pid')

  # The Block performing the DIC
  dis = crappy.blocks.DISCorrel('', display_images=True,
                                labels=['t(s)', 'meta', 'x', 'y',
                                        'measured_Exx(%)', 'measured_Eyy(%)'],
                                display_freq=True, iterations=0,
                                finest_scale=2,
                                image_generator=Apply_strain_img(img))
  crappy.link(machine, dis)

  # The PID block takes TWO inputs: the setpoint and the feedback
  # Here the setpoint is coming from the generator (target_Exx(%))
  # and the feedback is the strain measured with the DIC (measured_Exx(%))
  # The output is the command sent to the machine
  # In a real-world scenario, consider using out_min and out_max
  # to clamp the output value and i_limit to prevent over_integration.
  pid = crappy.blocks.PID(kp=0.5, ki=2, kd=0.05, target_label='target_Exx(%)',
                          input_label='measured_Exx(%)', send_terms=True)

  # We link the two inputs and the output
  # The order does not matter, the inputs are identified thanks to the
  # keywords target_label (setpoint) and input_label (feedback)
  crappy.link(generator, pid)
  crappy.link(dis, pid)
  crappy.link(pid, machine)

  # To see what is sent to the machine
  # Since send_terms is given to the PID block, it also returns the 3
  # channels of the PID (P, I and D). This can be useful when adjusting
  # the gains
  graph_pid = crappy.blocks.Grapher(('t(s)', 'pid'), ('t(s)', 'p_term'),
                                    ('t(s)', 'i_term'), ('t(s)', 'd_term'))
  crappy.link(pid, graph_pid, modifier=crappy.modifier.Mean(10))

  # To see the commanded and the measured strains
  graph_def2 = crappy.blocks.Grapher(('t(s)', 'measured_Exx(%)'),
                                     ('t(s)', 'target_Exx(%)'))
  crappy.link(dis, graph_def2, modifier=crappy.modifier.Mean(10))
  crappy.link(generator, graph_def2, modifier=crappy.modifier.Mean(10))

  crappy.start()
