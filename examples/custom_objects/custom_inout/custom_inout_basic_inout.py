# coding: utf-8

"""
This example demonstrates the instantiation of a custom InOut object in Crappy,
in the case when the InOut both acquires data from hardware and sets received
commands on it. The example presented here shows only the basic steps for
creating an InOut object that acquires and sets values. This example is an
extension of the custom_inout_basic_in.py example, so you should read it before
this one. It does not require any hardware nor specific Python module to run.

In Crappy, users can define their own InOut objects and use them along with
the IOBlock. This way, users can interface with their own hardware without
having to integrate it in the distributed version of Crappy.

Here, a very simple InOut object is instantiated for setting received commands.
It can also return data, so that the effect of the command can be read and
displayed to the user. This InOut is driven by an IOBlock, that receives
commands from a Generator and sends measured values to a Dashboard for display.
The Generator also sends its command to the Dashboard. The command sent to the
IOBlock and the measured one should match, with a small delay.

After starting this script, simply watch how the command values are
successfully sent to the IOBlock, set on the InOut, read back from the InOut,
and transmitted to the Dashboard. You can adjust the value of the max_value
setting on the IOBlock, and see how it is successfully set on the InOut and
modifies its output accordingly. This demo ends after 22s. You can also hit
CTRL+C to stop it earlier, but it is not a clean way to stop Crappy.
"""

import crappy
from typing import Optional
from time import time


class CustomInOut(crappy.inout.InOut):
  """This class demonstrates the instantiation of a custom InOut object in
  Crappy.

  It is fully recognized as an InOut, and can be used by any IOBlock. Each
  InOut must be a child of crappy.inout.InOut, otherwise it is not recognized
  as such.
  """

  def __init__(self, max_value: Optional[float] = None) -> None:
    """In this method you should initialize the Python objects that you will
    use in the class.

    You must initialize the parent class somewhere in this method. It is also
    where the arguments are passed to the InOut by the IOBlock.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    self._max_value = max_value if max_value is not None else float('inf')
    self._value: Optional[float] = None

  def open(self) -> None:
    """In this method you would perform any action needed to connect to the
    hardware, initialize it, and tune its settings.

    There is no action to perform in this simple demo though.
    """

    ...

  def get_data(self) -> dict[str, float]:
    """This method is used for acquiring data from the hardware.

    Here, it returns the current timestamp as well as the value of the _buffer
    value if it was already set.

    Note that unlike in the custom_inout_basic_in.py example, a dictionary is
    returned here so the labels are provided directly in this method. The
    labels given as arguments of the IOBlock are ignored.
    """

    if self._value is not None:
      return {'t(s)': time(), 'signal': self._value}

  def set_cmd(self, cmd: float) -> None:
    """This method is used for setting commands on hardware when the IOBlock
    has incoming Links.

    Here, it sets the _value buffer to the value of the received command. If
    the received value is greater than the maximum allowed value given as an
    argument, it sets the _value buffer to the maximum allowed value.
    """

    self._value = min(cmd, self._max_value)

  def close(self) -> None:
    """In this method you would perform any action needed to disconnect from
    the hardware and release the resources.

    There is no action to perform in this simple demo though.
    """

    ...


if __name__ == '__main__':

  # This Generator Block generates a sine wave and sends it to the IOBlock as
  # a command to set on the InOut
  gen = crappy.blocks.Generator(
      # Generating a sine wave of amplitude 2 and frequency 0.2 centered on 0
      ({'type': 'Sine', 'amplitude': 2, 'condition': 'delay=20', 
        'freq': 1 / 5},),
      cmd_label='cmd',  # Tha label carrying the generated signal
      freq=10,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This IOBlock drives the custom InOut object defined above. It receives the
  # commands to set from the Generator Block, and sets them on the InOut by
  # calling the set_cmd method. It also reads data from the InOut by calling
  # the get_data method, and sends the acquired values to the Dashboard for
  # display
  io = crappy.blocks.IOBlock(
      'CustomInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'blabla'),  # Will be ignored because the get_data
      # method returns a dictionary and not nameless values
      cmd_labels=('cmd',),  # The names of the labels carrying the commands to
      # set using the set_cmd method
      freq=30,  # Lowering the default frequency because it's just a demo
      max_value=0.5,  # This argument is not recognized by the IOBlock so it
      # will be passed to the InOut object

      # Sticking to default for the other arguments
  )

  # This Dashboard Block displays the data it receives from the IOBlock. It
  # prints the current timestamp and the value of the 'signal' label.
  dash = crappy.blocks.Dashboard(
      ('t(s)', 'signal', 'cmd'),  # The names of the labels to display
      freq=30,  # Useless to loop at a higher frequency than the IOBlock

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, io)
  crappy.link(io, dash)
  crappy.link(gen, dash)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
