# coding: utf-8

"""
This example demonstrates the instantiation of a custom Modifier object in
Crappy. It requires neither hardware nor any specific Python modules to run.

In Crappy, users can define their own Modifier objects and add them to Links
between Blocks. This lets users customize their test scripts precisely without
integrating new Modifiers into the distributed version of Crappy. Any callable,
including a function, can be used as a Modifier; this example uses a class.

Here, a Generator Block outputs a basic sine wave and sends it to a Dashboard
Block for display. On the Link between these two Blocks, a custom Modifier
object is added to modify the data on the fly. This Modifier calculates and
adds the RMS value to the transmitted data and removes specified labels from
it. The goal is to show the methods for creating a custom Modifier object.

After starting this script, watch how the data is modified by the custom
Modifier on its way to the Dashboard. The RMS value is successfully calculated
and added, while the 'index' label is removed. The RMS value gradually
stabilizes at sqrt(2)/2, as expected for a sine wave with no offset. This
demo ends after 22 seconds. Click the stop button to end the demo early.
"""

import crappy
from typing import TypeVar
from collections.abc import Sequence
from math import sqrt

T = TypeVar('T')


class CustomModifier(crappy.modifier.Modifier):
  """This class demonstrates the instantiation of a custom Modifier object in
  Crappy.

  A Modifier should preferably inherit from crappy.modifier.Modifier, but this
  is not mandatory. Any callable that takes and returns a dictionary can be
  used as a Modifier, including a simple function.

  Here, this custom Modifier calculates the RMS value of a given received label
  and adds it to the data. It also removes given labels from the transmitted
  data.
  """

  def __init__(self,
               rms_label: str,
               input_label: str,
               to_delete: Sequence[str] | None = None) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the Modifier.

    Args:
      rms_label: The label that will carry the calculated RMS value.
      input_label: The label to use for calculating the RMS value.
      to_delete: An iterable containing labels to remove from the data
        dictionary.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Instantiating attributes based on the given arguments
    self._rms_label = rms_label
    self._input_label = input_label
    self._to_delete = to_delete if to_delete is not None else list()

    # These attributes are used later for generating the RMS value
    self._square_sum = 0
    self._nb_samples = 0

  def __call__(self, data: dict[str, T]) -> dict[str, T]:
    """This method is the one that transforms the received data and returns the
    modified version.

    Args:
      data: This dictionary carries the original data that was sent through the
        Link. It can be modified at will.

    Returns:
      A dictionary containing the modified data, that is then transmitted to
      the downstream Block. If nothing is returned, nothing is transmitted.
    """

    # Removing the given labels from the received data
    for label in self._to_delete:
      data.pop(label)

    # Generating the RMS value and adding it to the received data
    self._nb_samples += 1
    self._square_sum += data[self._input_label] ** 2
    data[self._rms_label] = sqrt(self._square_sum / self._nb_samples)

    # Returning the modified data, otherwise nothing is transmitted to the
    # downstream Block
    return data


if __name__ == '__main__':

  # This Generator Block generates a sine wave and sends it to the Dashboard
  # Block for display. On its way to the Dashboard, the sent data is caught
  # and modified by the CustomModifier Modifier defined above
  gen = crappy.blocks.Generator(
      # Generating a sine wave of period 6 seconds and amplitude 2
      ({'type': 'Sine', 'amplitude': 2, 'freq': 1 / 6,
        'condition': 'delay=20'},),
      cmd_label='signal',  # The label carrying the generated signal
      path_index_label='index',  # The label carrying the current Path index
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Dashboard Block displays the data sent by the Generator Block and
  # modified by the CustomModifier on the way
  # You can see that the 'rms' label is successfully added, and that the
  # 'index' label is deleted
  dash = crappy.blocks.Dashboard(
      # The names of the labels whose values to display
      labels=('t(s)', 'signal', 'rms', 'index'),
      nb_digits=3,  # Displaying more digits than default to get a finer view
      # of the received values
      freq=30  # Useless to loop faster than the Generator Block

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, dash,
              # Instantiating here the CustomModifier defined above, with the
              # correct arguments
              modifier=CustomModifier(rms_label='rms',
                                      input_label='signal',
                                      to_delete=('index',)))

  # This Block provides a clean way to stop the test before it ends
  stop = crappy.blocks.StopButton()

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
