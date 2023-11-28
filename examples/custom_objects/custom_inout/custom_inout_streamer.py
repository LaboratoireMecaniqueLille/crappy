# coding: utf-8

"""
This example demonstrates the instantiation of a custom InOut object in Crappy,
in the case when the InOut only acquires data from hardware and only supports
the streamer mode. It shows a very basic example of such an InOut. It does not
require any hardware nor specific Python module to run.

In Crappy, users can define their own InOut objects and use them along with
the IOBlock. This way, users can interface with their own hardware without
having to integrate it in the distributed version of Crappy.

Here, a very simple InOut object is instantiated for reading data and returning
it as a stream using the get_stream method. It is driven by an IOBlock that
sends the acquired data to a Dashboard Block for display, with a Demux Modifier
on the Link that converts the stream to regular data. The acquired data is
simply randomly generated numbers. The goal here is to show the basic methods
to use for creating a custom InOut object that acquires data in streamer mode.
Note that in addition, a StopButton Block allows stopping the script properly
without using CTRL+C by clicking on a button.

After starting this script, simply watch how the data is successfully generated
by the InOut object, transmitted by the IOBlock and displayed by the Dashboard
Block. To end this demo, click on the stop button that appears. You can also
hit CTRL+C, but it is not a clean way to stop Crappy.
"""

import crappy
import numpy as np
import numpy.random as rd
from typing import Tuple
from time import time


class CustomInOut(crappy.inout.InOut):
  """This class demonstrates the instantiation of a custom InOut object in
  Crappy for use in streamer mode.

  It is fully recognized as an InOut, and can be used by any IOBlock. Each
  InOut must be a child of crappy.inout.InOut, otherwise it is not recognized
  as such.
  """

  def __init__(self) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the InOut by the IOBlock.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

  def open(self) -> None:
    """In this method you would perform any action needed to connect to the
    hardware, initialize it, and tune its settings.

    There is no action to perform in this simple demo though.
    """

    ...

  def start_stream(self) -> None:
    """In this method you would perform any action required for starting the
    acquisition of the stream on hardware.

    There is no action to perform in this simple demo though.
    """

    ...

  def get_stream(self) -> Tuple[np.ndarray, np.ndarray]:
    """This method acquires the stream data and returns it in one array
    containing the time information and another array containing the data.

    Here, 10 random values are generated and returned.
    """

    t = time()
    t_arr = np.array([t + i * 0.005 for i in range(10)])
    val_arr = rd.random((10, 1))
    return t_arr, val_arr

  def stop_stream(self) -> None:
    """In this method you would perform any action required for stopping the
    acquisition of the stream on hardware.

    There is no action to perform in this simple demo though.
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
  # only outgoing Links, and it is in streamer mode, it only reads data from
  # the InOut by calling the get_stream method. This data is then sent to the
  # Dashboard Block for display, with a Demux Modifier on the Link to convert
  # the stream to regular data
  io = crappy.blocks.IOBlock(
      'CustomInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'stream'),  # The names of the labels to send to the
      # downstream Block
      freq=20,  # Lowering the default frequency because it's just a demo
      streamer=True,  # Switches the IOBlock to streamer mode

      # Sticking to default for the other arguments
  )

  # This Dashboard Block displays the data it receives from the IOBlock and
  # converted by the Demux Modifier. It prints the current timestamp and the
  # value of the 'signal' label.
  dash = crappy.blocks.Dashboard(
      ('t(s)', 'signal'),  # The names of the labels to display
      freq=20,  # Useless to loop at a higher frequency than the IOBlock

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(io, dash,
              # This Demux Modifier is mandatory to convert the stream data to
              # regular data readable by the Dashboard Block
              modifier=crappy.modifier.Demux(labels=('signal',),
                                             stream_label='stream'))

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
