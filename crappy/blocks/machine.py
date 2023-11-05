# coding: utf-8

from time import time
from typing import Dict, List, Any, Optional
from warnings import warn

from .block import Block
from ..actuator import actuator_list


class Machine(Block):
  """This block is meant to drive one or several :ref:`Actuators`.

  The possibility to drive several Actuators from a unique block is given so
  that they can be driven in a synchronized way. If synchronization is not
  needed, it is preferable to drive the Actuators from separate Machine blocks.
  """

  def __init__(self,
               actuators: List[Dict[str, Any]],
               common: Optional[Dict[str, Any]] = None,
               time_label: str = 't(s)',
               spam: bool = False,
               freq: float = 200,
               verbose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      actuators: The :obj:`list` of all the :ref:`Actuators` this block needs
        to drive. It contains one :obj:`dict` for every Actuator, with
        mandatory and optional keys. The keys providing information on how to
        drive the Actuator are listed below. Any other key will be passed to
        the Actuator object as argument when instantiating it.
      common: The keys of this :obj:`dict` will be common to all the Actuators.
        If it conflicts with an existing key for an Actuator, the common one
        will prevail.
      time_label: If reading speed or position from one or more Actuators, the
        time information will be carried by this label.
      spam: If :obj:`True`, a command is sent to the Actuators on each loop of
        the block, else it is sent every time a command is received.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, prints the looping frequency of the block.

    Note:
      - ``actuators`` keys:

        - ``type``: The name of the Actuator class to instantiate.
        - ``cmd``: The label carrying the command for driving the Actuator.
        - ``mode``: Can be either `'speed'` or `'position'`. Will either call
          :meth:`set_speed` or :meth:`set_position` to drive the actuator, and
          :meth:`get_speed` or :meth:`get_position` for acquiring the current
          speed or position.
        - ``speed``: If mode is `'position'`, the speed at which the Actuator
          should move. This key is not mandatory, even in the `'position'`
          mode.
        - ``pos_label``: If given and the mode is `'position'`, the block will
          return the value of :meth:`get_position` under this label. This key
          is not mandatory.
        - ``speed_label``: If given and the mode is `'speed'`, the block will
          return the value of :meth:`get_speed` under this label. This key is
          not mandatory.
    """
    
    if verbose:
      warn("The verbose argument will be replaced by display_freq and debug "
           "in version 2.0.0", FutureWarning)

    super().__init__()
    self.freq = freq
    self.verbose = verbose

    if common is None:
      common = dict()

    self._time_label = time_label
    self._spam = spam

    self._settings = list()

    # For each actuator, parsing the given settings
    for actuator in actuators:
      actuator.update(common)
      settings = dict()

      # Getting all the possible arguments from the actuator dict
      for arg in ('type', 'cmd', 'mode', 'speed', 'pos_label', 'speed_label'):
        try:
          settings[arg] = actuator.pop(arg)
        except KeyError:
          # If an arg is not given, setting it to None
          settings[arg] = None

      # Putting all the remaining settings together under the kwargs key
      settings['kwargs'] = actuator

      # Making sure that the mandatory arguments are given
      for arg in ('type', 'cmd', 'mode'):
        if settings[arg] is None:
          raise ValueError(f"An actuator given as argument of the Machine "
                           f"block doesn't define the {arg} setting !")

      # Making sure that the given mode is valid
      if settings['mode'].lower() not in ('position', 'speed'):
        raise ValueError(f"The 'mode' setting for the actuators should be "
                         f"either 'position' or 'speed' !")

      # Storing the settings dict
      self._settings.append(settings)

    # Instantiating the actuators
    self._actuators = [actuator_list[settings['type'].capitalize()]
                       (**settings['kwargs']) for settings in self._settings]

  def prepare(self) -> None:
    """Checks the validity of the linking and initializes all the Actuator
    objects to drive."""

    # Checking the consistency of the linking
    if not self.inputs and not self.outputs:
      raise IOError("The Machine block is neither an input nor an output !")

    # Opening each actuator
    for actuator in self._actuators:
      actuator.open()

  def loop(self) -> None:
    """Receives the commands from upstream blocks, sets them on the actuators
    to drive, and sends the read positions and speed to the downstream
    blocks."""

    # Receiving the latest command
    if self._spam:
      recv = self.get_last(blocking=False)
    else:
      recv = self.recv_all_last()

    # Iterating over the actuators for setting the commands
    if recv:
      for actuator, settings in zip(self._actuators, self._settings):
        mode, cmd = settings['mode'], settings['cmd']

        # If a command was received, setting it
        if cmd in recv:
          if mode == 'speed':
            actuator.set_speed(recv[cmd])
          elif mode == 'position':
            actuator.set_position(recv[cmd], settings['speed'])

    to_send = {}

    # Iterating over the actuators to get the speeds and the positions
    for actuator, settings in zip(self._actuators, self._settings):
      pos_label, speed_label = settings['pos_label'], settings['speed_label']
      mode = settings['mode']
      if mode == 'position' and pos_label is not None:
        position = actuator.get_position()
        if position is not None:
          to_send[pos_label] = position
      elif mode == 'speed' and speed_label is not None:
        speed = actuator.get_speed()
        if speed is not None:
          to_send[speed_label] = speed

    # Sending the speed and position values if any
    if to_send:
      to_send[self._time_label] = time() - self.t0
      self.send(to_send)

  def finish(self) -> None:
    """Stops and closes all the actuators to drive."""

    for actuator in self._actuators:
      actuator.stop()
    for actuator in self._actuators:
      actuator.close()
