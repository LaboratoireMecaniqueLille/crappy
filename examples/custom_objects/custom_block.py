# coding: utf-8

"""
This example demonstrates the instantiation of a custom Block object in Crappy.
The example presented here shows most of the attributes and methods to use when
building a Block. It does not require any hardware nor specific Python module
to run.

In Crappy, users can define their own Block object and use them in their
scripts. This way, users can finely customize their test scripts without having
to integrate new Blocks in the distributed version of Crappy.

Here, a new Block object is instantiated that simply displays in the console
the data it receives from a Generator Block. It also sends data for a Dashboard
Block to display. Several arguments of the custom Block can be tuned, including
one allowing to choose which method to use for receiving data. The goal here is
to show the methods and attributes to use for creating a custom Block object.

After starting this script, watch in the console how the data from the
Generator is received by the CustomBlock. In the Dashboard window you can also
see how the data is successfully sent by the CustomBlock. You can adjust the
various settings of the Block, especially the method to use for receiving data.
Also play with the looping frequency of the custom and Generator Blocks to see
what happens when several chunks are waiting in the Links. This demo ends after
t_limit seconds. You can also hit CTRL+C to stop it earlier, but it is not a
clean way to stop Crappy.
"""

import crappy
from collections.abc import Sequence
import logging
from time import sleep, time


class CustomBlock(crappy.blocks.Block):
  """This class demonstrates the instantiation of a custom Block object in
  Crappy.

  It shows most of the methods and attributes that can be used when building
  your own Blocks. Because it is a child class of crappy.blocks.Block, this
  class is fully recognized as such and can be used in a Crappy script.
  """

  def __init__(self,
               recv_meth: str,
               labels: Sequence[str],
               t_limit: float = float('inf'),
               freq: float | None = 100,
               display_freq: bool | None = False) -> None:
    """This method performs several critical actions.

    First, it initializes the parent class. Then, it allows to set several
    special attributes that control the way the Block runs. And finally, it
    is where the arguments are passed to the Block and where they should be
    handled.

    Args:
      recv_meth: The method to use for receiving data from upstream Blocks.
        Should be one of 'data', 'last_data', 'all_data', 'all_data_raw'.
      labels: The labels to use for sending data to downstream Blocks.
      t_limit: The Block will stop itself and the entire Crappy test after that
        many seconds. Set to None to keep the Block running forever.
      freq: The target looping frequency for the Block.
      display_freq: Set to True to display the achieved looping frequency as a
        log message.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # These attributes are special ones that affect the behavior of the Block

    # The niceness attribute can be used on Linux to increase or decrease the
    # priority of the Block over other processes in case the CPU is used at
    # 100%
    self.niceness: int = 0

    # The label attribute stores the labels to associate values to if they are
    # given raw to the send method and not in a dictionary
    self.labels = labels

    # The freq attribute indicates the target looping frequency for the Block
    self.freq = freq

    # When the display_freq attribute is set to True, the achieved looping
    # frequency of the Block is displayed every 2s
    self.display_freq = display_freq

    # When the debug attribute is set to True, the logging level for this Block
    # switches from INFO to DEBUG
    self.debug = False

    # Setting these attributes to reuse them later
    self._recv_meth = recv_meth
    self._count = 0
    self._t_limit = t_limit

  def prepare(self) -> None:
    """This method is called before the Block starts looping.

    Crappy waits for all the Blocks to have completed their prepare method
    before allowing them to loop. This method should be used for performing
    initialization tasks, like connecting to hardware. It is also where you
    should perform tasks affected by multiprocessing, and not in __init__.

    Here, a message is displayed to show that the method was called.
    """

    # To display text messages, use the self.log method rather than print
    self.log(logging.WARNING, "The prepare method is called before the Blocks "
                              "start looping")
    sleep(1)

  def begin(self) -> None:
    """This method is called once, as the first loop of the Block.

    It should be used to perform actions that should happen only once at the
    very beginning of the test, but later than the prepare method. For example,
    you can send here a specific value to the downstream Blocks.

    Here, a specific message is sent to downstream Blocks. Because the begin
    method is not affected by the frequency regulation, the sleep function is
    called to avoid switching right away to the first loop.
    """

    self.log(logging.WARNING, "If defined, the begin method is called only "
                              "once, as the first loop")
    # The self.send method is used for sending values to downstream Blocks
    # Here, unlabeled values are given to send, so they will be labeled
    # automatically using the values given in the self.labels attribute
    self.send((time() - self.t0, "begin"))
    sleep(1)

  def loop(self) -> None:
    """The loop method is the core of the Block, that is called repeatedly
    during the test.

    In this method, you can receive incoming data from upstream Block and/or
    send data to downstream Blocks. You can also perform any other calculation,
    operation on files, interaction with a GUI, etc.

    Here, the various methods available for users to call are demonstrated. In
    particular, the method specified by the recv_meth argument is used for
    receiving data.
    """

    # The self.data_available method indicates whether new data points have
    # been received from upstream Blocks. It can be useful to stop the loop
    # early if nothing can be done without incoming data
    available = self.data_available()
    self.log(logging.WARNING, f"There is {'no' if not available else 'some'} "
                              f"data to read")

    # There are four possible ways to read data in Crappy. Each of the four
    # methods has its specificities, refer to the documentation for an
    # extensive description

    if self._recv_meth == 'data':
      # This method only reads the first available chunk (the oldest one) of
      # each incoming Link, and returns all the received labels in a single
      # dictionary. There might still be data left in the Links after this call
      data = self.recv_data()

    elif self._recv_meth == 'last_data':
      # This method reads all the available data from each incoming Link, but
      # returns only the latest values of each labels in a single dictionary.
      # This call flushes all the Links
      data = self.recv_last_data()

    elif self._recv_meth == 'all_data':
      # This method reads all the available data from each incoming Link, and
      # returns all the received values as lists in a single dictionary. This
      # call flushes all the Links
      data = self.recv_all_data()

    elif self._recv_meth == 'all_data_raw':
      # This method reads all the available data from each incoming Link, and
      # returns a list containing for each Link the dictionary that
      # recv_all_data would return if there was only one Link. This call
      # flushes all the Links
      data = self.recv_all_data_raw()

    else:
      raise ValueError(f"Unknown receive method given : {self._recv_meth}")

    # Displaying the received data to show what the receive methods do
    self.log(logging.WARNING, f"Received the following data : {data}")

    # Stopping the Block if the timeout is exceeded
    if time() - self.t0 > self._t_limit:
      self.log(logging.WARNING, "Calling the stop method because the timeout "
                                "is exceeded")
      # The self.stop method allows to stop the execution of the Block without
      # raising an exception. As a consequence, it also stops all the running
      # Blocks and then the entire Crappy script
      self.stop()

    # Sending the loop count to the downstream Blocks as an example of the
    # self.send method
    self._count += 1
    self.send((time() - self.t0, f'loop n°{self._count}'))

  def finish(self) -> None:
    """This method is called at the very end of the test.

    It is almost always called, even if an error occurred. You can use it to
    perform any task needed at termination, like disconnecting from hardware or
    releasing resources.

    Here, a message is displayed to show that the method was called.
    """

    self.log(logging.WARNING,
             "The finish method is always called no matter how the test "
             "ended, except if Crappy crashes really hard")
    sleep(1)


if __name__ == '__main__':

  # This Generator Block generates a sine wave and sends it to the CustomBlock
  gen = crappy.blocks.Generator(
      # generating a sine wave of amplitude 2 and period 10s
      ({'type': 'Sine', 'freq': 0.1, 'amplitude': 2, 'condition': None},),
      cmd_label='signal',  # The label carrying the generated signal
      freq=0.25,  # Setting a very low frequency to show how the CustomBlock
      # might sometimes not receive any data. Set to higher frequencies and
      # combine with varius receive methods to explore different behaviors

      # Sticking to default for the other arguments
  )

  # Instantiating here the CustomBlock defined above, and supplying the desired
  # arguments. This Block receives data from the Generator Block and sends its
  # output to the Dashboard Block
  custom = CustomBlock(
      recv_meth='data',  # Change to another valid method to see what changes
      labels=('t(s)', 'msg'),  # the labels carrying the data to send to the
      # downstream Block
      t_limit=30,  # The Block will stop after that many seconds, remove to let
      # the test run forever
      freq=0.5,  # Setting a low frequency to avoid spamming the console with
      # too many messages
      display_freq=False  # This argument is only for demonstration, it has no
      # effect since logging of INFO messages is disabled
  )

  # This Dashboard Block displays the data it receives from the CustomBlock
  dash = crappy.blocks.Dashboard(
      ('t(s)', 'msg'),  # The names of the labels to display

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, custom)
  crappy.link(custom, dash)

  # Mandatory line for starting the test, this call is blocking
  # Restraining the log level to WARNING so that only the messages from the
  # CustomBlock are displayed
  crappy.start(log_level=logging.WARNING)
