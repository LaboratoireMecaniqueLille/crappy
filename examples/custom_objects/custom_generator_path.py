# coding: utf-8

"""
This example demonstrates the instantiation of a custom Generator Path object
in Crappy. The example presented here shows most of the features of a Path
object. It does not require any hardware to run, but necessitates the
matplotlib Python module to be installed.

In Crappy, users can define their own Path objects and use them along with
the Generator Block. This way, users can finely customize their test scripts
without having to integrate new Paths in the distributed version of Crappy.

Here, a new Path object is instantiated that generates sine waves elevated to
a given power. This behavior could also be achieved using the Sine Path and a
Modifier, but the custom Path is defined here for the sake of the demo. The
Path is controlled by a Generator Block, that outputs its data to a Grapher
Block for display. The goal here is to show the methods to use for creating a
custom Path object.

After starting this script, watch how the signal is generated as desired by the
custom Path and displayed in the Grapher. You can adjust the various settings
of the Path in the Generator Block and see how it affects the signal. This demo
ends after 22s. You can also hit CTRL+C to stop it earlier, but it is not a
clean way to stop Crappy.
"""

import crappy
from crappy.blocks.generator_path.meta_path import Path, ConditionType
from math import pi, sin, copysign
from time import time


class CustomPath(Path):
  """This class demonstrates the instantiation of a custom Generator Path
  object in Crappy.

  It is fully recognized as a Path, and can be used by any Generator Block.
  Each Path must be a child of crappy.blocks.generator_path.meta_path.Path,
  otherwise it is not recognized as such.
  """

  def __init__(self,
               amplitude: float,
               freq: float,
               power: int,
               condition: str | ConditionType | None,
               offset: float | None = 0.0) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the Path by the Generator Block.

    Args:
      amplitude: The peak to peak amplitude of the sine wave.
      freq: The frequency of the sine wave in Hz.
      power: The power to elevate the sine wave to, as an integer.
      condition: The stop condition for this Generator Path.
      offset: The offset of the sine wave.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # The parse_condition method allows to parse a condition given as a string
    # or as a callable or None, and returns a callable that can be used later
    # to check whether a stop condition is met
    self._condition = self.parse_condition(condition)

    # Defining these useful attributes based on the given arguments
    self._k = 2 * pi * freq
    self._power = power
    self._amplitude = amplitude / 2
    self._offset = offset

  def get_cmd(self, data: dict[str, list]) -> float:
    """This method is the one that generates and returns the value to output
    for the Generator Block.

    It is also in charge of checking whether the stop conditions are met, if
    the Path features any.

    Args:
      data: This dictionary contains all the data received by the Generator
        Block since its last loop. It can be used to check if a stop condition
        is met, or for any other purpose.

    Returns:
      This method must return a value as a float, that will be sent to
      downstream Blocks by the Generator Block.
    """

    # self._condition is callable, and takes as argument the dictionary of the
    # data received by the Generator Block since the last loop
    # It returns True if the stop condition is met, in which case a
    # StopIteration exception must be raised
    if self._condition(data):
      raise StopIteration

    # Getting the sine value, elevating it to the right power and returning it
    # The t0 attribute stores the timestamp of the beginning of the current
    # Path
    # The last_cmd attribute stores the last value sent by the Generator Block
    # before switching to the current Path. It is not used here
    sine = sin((time() - self.t0) * self._k)
    power = copysign(sine ** self._power, sine)
    return power * self._amplitude + self._offset


if __name__ == '__main__':

  # This Generator Block generates a signal using the CustomPath defined above
  # It then sends it to the Grapher Block for display
  # The generated signal is a sine wave elevated to a given power (integer),
  # and the parameters of the sine can be tuned here
  gen = crappy.blocks.Generator(
      ({'type': 'CustomPath',  # The signal is generated using the CustomPath
        # defined above
        'amplitude': 2,  # The peak to peak amplitude of the sine is 2
        'freq': 0.5,  # The period of the sine wave is 2s
        'power': 2,  # The sine is elevated to the power 2
        'offset': 3,  # The sine is generated with an offset of 3
        'condition': 'delay=20'},),  # The Path stops after 20s
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the generated signal

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the custom signal it receives from the
  # Generator Block
  graph = crappy.blocks.Grapher(
      ('t(s)', 'signal'),  # The names of the labels to display

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
