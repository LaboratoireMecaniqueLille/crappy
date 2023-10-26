# coding: utf-8

"""
This example demonstrates the instantiation of a custom Actuator object in
Crappy, in the case when the Actuator is driven in position. The example
presented here shows only the basic steps for creating an Actuator object. It
does not require any hardware to run, but necessitates the matplotlib Python
module to be installed.

In Crappy, users can define their own Actuator objects and use them along with
the Machine Block. This way, users can interface with their own hardware
without having to integrate it in the distributed version of Crappy.

Here, a very simple Actuator object is instantiated. It can be driven in
position, and output its current speed and position. It is controlled by a
Machine Block, that sets the target position commands it receives from a
Generator Block. The Generator and the Machine Block send respectively the
target and measured positions to a Grapher Block for display. The goal here is
to show the basic methods to use for creating a custom Actuator object driven
in position mode. There is no example of a custom Actuator driven in speed
mode, but it is very similar to the position mode.

After starting this script, watch how the target position is set on the
Actuator and how its position is simultaneously acquired. Notice how the
position of the Actuator evolves towards the target. This demo ends after 42s.
You can also hit CTRL+C to stop it earlier, but it is not a clean way to stop
Crappy.
"""

import crappy
from typing import Optional
from time import time
from math import copysign


class CustomActuator(crappy.actuator.Actuator):
  """This class demonstrates the instantiation of a custom Actuator object in
  Crappy.

  It is fully recognized as an Actuator, and can be used by any Machine Block.
  Each Actuator must be a child of crappy.actuator.Actuator, otherwise it is
  not recognized as such.
  """

  def __init__(self, init_speed: float = 1) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the Actuator by the Machine Block.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # These attributes keep track of the current state of the Actuator
    self._speed: float = init_speed
    self._pos: float = 0
    self._target_pos: float = 0
    self._last_t: float = time()

  def open(self) -> None:
    """In this method you would perform any action needed to connect to the
    hardware, initialize it, and tune its settings.

    Here, we just update the last timestamp to have it as close as possible to
    the actual start of the test.
    """

    self._last_t: float = time()

  def set_speed(self, speed: float) -> None:
    """This method is used for setting a target speed value on the Actuator.

    Here, it is not defined as this class is not meant to be driven in speed.
    Otherwise, it would normally communicate with hardware.
    """

    ...

  def set_position(self, position: float, speed: Optional[float]) -> None:
    """This method is used for setting a target position value on the Actuator.

    Along with the target position value, a speed argument is always received.
    If a value is specified, it is received as a float. If no value is
    specified, it is set to None.

    Here, this method only sets the internal attributes according to the
    received input. In a real-life Actuator, this method would certainly
    communicate with hardware.
    """

    if speed is not None:
      self._speed = speed

    self._target_pos = position

  def get_speed(self) -> float:
    """This method should acquire and return the current speed of the Actuator.

    Here, it first updates the state variables then returns the current speed
    value. In a real-life Actuator, this method would certainly communicate
    with hardware.
    """

    self._update()
    return self._speed

  def get_position(self) -> float:
    """This method should acquire and return the current position of the
    Actuator.

    Here, it first updates the state variables then returns the current
    position value. In a real-life Actuator, this method would certainly
    communicate with hardware.
    """

    self._update()
    return self._pos

  def close(self) -> None:
    """In this method you would perform any action needed to disconnect from
    the hardware and release the resources.

    There is no action to perform in this simple demo though.
    """

    ...

  def _update(self) -> None:
    """This method updates the current position and speed of the Actuator, for
    the get_speed and get_position methods to return."""

    # First updating the time information
    t = time()
    delta = self._speed * (t - self._last_t)
    self._last_t = t

    # Only updating the position if the target is not reached yet
    if self._pos != self._target_pos:
      # If we're close enough to the target, just stabilize at it
      if abs(self._target_pos - self._pos) <= delta:
        self._pos = self._target_pos
      # Otherwise, just moving closer to it
      else:
        self._pos += copysign(delta, self._target_pos - self._pos)


if __name__ == '__main__':

  # This Generator Block generates the target position signal and sends it to
  # the Machine Block that sets it on the CustomActuator. It also sends the
  # signal to the Grapher Block for display.
  gen = crappy.blocks.Generator(
      # Generating a not-so-basic Path to demonstrate how the Actuator reacts
      # to changes in the target position
      ({'type': 'Constant', 'value': 10, 'condition': 'delay=10'},
       {'type': 'Ramp', 'speed': 2, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 20, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 0, 'condition': 'delay=20'}),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='target(mm)',  # The label carrying the generated signal
      spam=True,  # Sending value at each loop for a nice display on the graph

      # Sticking to default for the other arguments
  )

  # This Machine Block drives the CustomActuator Actuator defined above. It
  # sets the target position received from the Generator Block, measures the
  # current position on the Actuator, and sends it to the Grapher Block for
  # display
  mot = crappy.blocks.Machine(
      ({'type': 'CustomActuator',  # The name of the Actuator to drive
        'mode': 'position',  # Driving in position mode, not in speed mode
        'cmd_label': 'target(mm)',  # The label carrying the target position
        'position_label': 'pos(mm)',  # The label carrying the current position
        'speed': 1.5,  # Setting a fixed speed of 1.5 for the entire test
        'init_speed': 0},),  # This argument is not recognized by the Machine
      # Block so it will be passed as an argument to the Actuator
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays both the target position it receives from the
  # Generator Block and the measured position from the Machine Block
  # It is clearly visible how the CustomActuator tries its best to reach the
  # target position and stabilizes once it reaches it
  graph = crappy.blocks.Grapher(
      # The names of the labels to display
      ('t(s)', 'target(mm)'), ('t(s)', 'pos(mm)'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)
  crappy.link(gen, mot)
  crappy.link(mot, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
