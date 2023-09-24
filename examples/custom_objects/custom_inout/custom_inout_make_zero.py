# coding: utf-8

"""
This example demonstrates the instantiation of a custom InOut object in Crappy,
in the case when the make_zero method is customized by the user. The example
presented here is almost the same as the custom_inout_basic_in.py, except the
make_zero method is also defined in the custom InOut. It does not require any
hardware nor specific Python module to run.

In Crappy, users can define their own InOut objects and use them along with
the IOBlock. This way, users can interface with their own hardware without
having to integrate it in the distributed version of Crappy.

Here, a very simple InOut object is instantiated for reading data. It is driven
by an IOBlock that sends the acquired data to a Dashboard Block for display.
The InOut object returns random data centered on an adjustable offset value.
Because the make_zero_delay argument of the IOBlock is given, values are
acquired before the test starts. While the normal behavior would be for these
values to be used for offsetting the signal to 0, the behavior is here modified
by supplying a custom make_zero method in the custom InOut. The goal here is to
show how to define a custom make_zero method for InOut objects. Note that in
addition, a StopButton Block allows stopping the script properly without using
CTRL+C by clicking on a button.

After starting this script, observe how the given offset is inverted instead of
being compensated to 0. You can modify the delay of the acquisition, or the
value of the offset, and see how the behavior persists. To end this demo, click
on the stop button that appears. You can also hit CTRL+C, but it is not a clean
way to stop Crappy.
"""

import crappy
from random import random
from typing import Tuple, Optional
from time import time


class CustomInOut(crappy.inout.InOut):
  """This class demonstrates the instantiation of a custom InOut object in
  Crappy.

  It is fully recognized as an InOut, and can be used by any IOBlock. Each
  InOut must be a child of crappy.inout.InOut, otherwise it is not recognized
  as such.
  """

  def __init__(self, offset: Optional[float] = None) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the InOut by the IOBlock.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Instantiating the attribute
    self._offset: float = 0 if offset is None else offset

  def open(self) -> None:
    """In this method you would perform any action needed to connect to the
    hardware, initialize it, and tune its settings.

    There is no action to perform in this simple demo though.
    """

    ...

  def get_data(self) -> Tuple[float, float]:
    """This method is used for acquiring data from the hardware.

    Here, it returns the current timestamp and a random value in the interval
    [_offset, _offset + 1).
    """

    # Returning the timestamp and a random value
    return time(), self._offset + random()

  def set_cmd(self, *_) -> None:
    """This method is used for setting commands on hardware when the IOBlock
    has incoming Links.

    It is thus not used in this demo and does not need to be defined.
    """

    ...

  def close(self) -> None:
    """In this method you would perform any action needed to disconnect from
    the hardware and release the resources.

    There is no action to perform in this simple demo though.
    """

    ...

  def make_zero(self, delay: float) -> None:
    """By overwriting this method you can customize the behavior of the zeroing
    when the make_zero_delay argument of the IOBlock is set.

    For example, you can retrieve the compensation values calculated by the
    original make_zero method and set them directly as an offset on hardware if
    that is supported. The compensation values can be accessed in the
    _compensations attribute.

    Here, to demonstrate the possibility of customizing the behavior, the
    compensation values are simply multiplied by 2.
    """

    # Acquiring the values and calculating the compensations
    super().make_zero(delay)

    # Multiplying the compensations by 2 as a custom behavior
    self._compensations = [2 * comp for comp in self._compensations]


if __name__ == '__main__':

  # This IOBlock drives the custom InOut object defined above. Because it has
  # only outgoing Links, it only reads data from the InOut by calling the
  # get_data method. This data is then sent to the Dashboard Block for display
  io = crappy.blocks.IOBlock(
      'CustomInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'signal'),  # The names of the labels to send to the
      # downstream Block
      freq=30,  # Lowering the default frequency because it's just a demo
      offset=10,  # This argument is not recognized by the IOBlock so it will
      # be passed to the InOut object
      make_zero_delay=2,  # Mandatory argument for the make_zero behavior to
      # be activated

      # Sticking to default for the other arguments
  )

  # This Dashboard Block displays the data it receives from the IOBlock. It
  # prints the current timestamp and the value of the 'signal' label.
  dash = crappy.blocks.Dashboard(
      ('t(s)', 'signal'),  # The names of the labels to display
      freq=30,  # Useless to loop at a higher frequency than the IOBlock

      # Sticking to default for the other arguments
  )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(io, dash)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
