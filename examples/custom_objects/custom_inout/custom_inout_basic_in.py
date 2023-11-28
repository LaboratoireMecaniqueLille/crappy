# coding: utf-8

"""
This example demonstrates the instantiation of a custom InOut object in Crappy,
in the case when the InOut only acquires data from hardware. The example
presented here shows only the basic steps for creating an InOut object that
acquires data. See the custom_inout_basic_inout.py for an example of InOut
setting commands on hardware. It does not require any hardware nor specific
Python module to run.

In Crappy, users can define their own InOut objects and use them along with
the IOBlock. This way, users can interface with their own hardware without
having to integrate it in the distributed version of Crappy.

Here, a very simple InOut object is instantiated for reading data. It is driven
by an IOBlock that sends the acquired data to a Dashboard Block for display.
The InOut object simply returns the index of its current loop, and has one
parameter that can be tuned. The goal here is to show the basic methods to
use for creating a custom InOut object that acquires data. Note that in
addition, A StopButton Block allows stopping the script properly without using
CTRL+C by clicking on a button.

After starting this script, simply watch how the data is successfully generated
by the InOut object, transmitted by the IOBlock and displayed by the Dashboard
Block. You can adjust the value of the max_value setting in the IOBlock and see
how it is successfully set on the InOut. To end this demo, click on the stop
button that appears. You can also hit CTRL+C, but it is not a clean way to stop
Crappy.
"""

import crappy
from typing import Tuple, Optional
from time import time


class CustomInOut(crappy.inout.InOut):
  """This class demonstrates the instantiation of a custom InOut object in
  Crappy.

  It is fully recognized as an InOut, and can be used by any IOBlock. Each
  InOut must be a child of crappy.inout.InOut, otherwise it is not recognized
  as such.
  """

  def __init__(self, max_value: Optional[int] = None) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the InOut by the IOBlock.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Instantiating the attributes
    self._count: int = 0
    self._max_val: int = max_value if max_value is not None else 1

  def open(self) -> None:
    """In this method you would perform any action needed to connect to the
    hardware, initialize it, and tune its settings.

    There is no action to perform in this simple demo though.
    """

    ...

  def get_data(self) -> Tuple[float, int]:
    """This method is used for acquiring data from the hardware.

    Here, it simply updates the loop counter and returns its value along with a
    timestamp. In a real-life InOut, it would likely read the value directly
    from hardware.
    """

    # Updating the loop counter
    self._count += 1

    # Returning the timestamp and the counter modulo the maximum allowed value
    return time(), self._count

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


if __name__ == '__main__':

  # This IOBlock drives the custom InOut object defined above. Because it has
  # only outgoing Links, it only reads data from the InOut by calling the
  # get_data method. This data is then sent to the Dashboard Block for display
  io = crappy.blocks.IOBlock(
      'CustomInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'signal'),  # The names of the labels to send to the
      # downstream Block
      freq=30,  # Lowering the default frequency because it's just a demo
      max_value=100,  # This argument is not recognized by the IOBlock so it
      # will be passed to the InOut object

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
