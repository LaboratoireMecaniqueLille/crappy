# coding: utf-8

from time import time, sleep
from typing import Dict, List, Any, Optional, Iterator
from itertools import cycle
from copy import deepcopy
import logging

from .meta_block import Block
from .generator_path.meta_path import paths_dict
from .._global import GeneratorStop


class GeneratorNoStop(Exception):
  """A custom exception for handling the case when the Generator should not
  raise a :exc:`CrappyStop` exception when it terminates."""


class Generator(Block):
  """This block generates a signal following a user-defined path.

  One Generator block can only generate one signal. Use multiple blocks if
  several signals are needed. Note that the default behavior of a Generator
  is to stop the entire script when it ends.

  It should be used as an input for other blocks, for example for driving a
  :ref:`Machine` or an :ref:`IOBlock`. It can also accept inputs, to make the
  path dependent on data coming from other blocks.
  """

  def __init__(self,
               path: List[Dict[str, Any]],
               freq: Optional[float] = 200,
               cmd_label: str = 'cmd',
               cycle_label: str = 'cycle',
               repeat: bool = False,
               spam: bool = False,
               display_freq: bool = False,
               end_delay: Optional[float] = 2,
               safe_start: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      path: It must be a :obj:`list` of :obj:`dict`, each dict providing the
        parameters to generate the path. Refer to the Note below for more
        information.
      freq: The looping frequency this block will try to achieve. Note that the
        higher this value, the more accurate the path will be. It will also
        consume more resources.
      cmd_label: The label of the signal sent to the downstream blocks.
      cycle_label: In addition to the `cmd_label`, this label holds the index
        of the current dict in the ``path`` list. Useful to trig a block  upon
        change in the current dict.
      repeat: If :obj:`True`, the ``path`` will loop forever instead of
        stopping when the list of dicts is exhausted.
      spam: If :obj:`True`, the signal value will be sent on each loop. Else,
        it will only be sent if it is different from the previous or if the
        ``path`` switched to a new dict.
      display_freq: if :obj:`True`, displays the loop frequency of the block
        and a message when switching to the next dict of ``path``.
      end_delay: When all the dicts in ``path`` are exhausted, waits this many
        seconds before stopping the entire program. Can be set to :obj:`None`,
        in which case the Generator won't stop the program when finishing.
      safe_start: Ensures the first dict in ``path`` waits for at least one
        data point from upstream blocks before sending the first value of the
        signal. Otherwise, the first value might be sent without checking the
        associated condition if its depends on labels from other blocks.

    Note:
      The different types of signals that can be generated by the Generator
      can be found at `generator path`. The ``path`` list contains one dict per
      signal shape to generate. They are generated in the order in which they
      appear in the list.

      Each dict contains information on the signal shape to generate, like its
      type, any applicable parameter(s), and the stop condition(s). Refer to
      the documentation of each signal shape to which information to give.
    """

    super().__init__()

    # Instantiating a few attributes based on the arguments
    self.niceness = -5
    self.freq = freq
    self.display_freq = display_freq
    self.labels = ['t(s)', cmd_label, cycle_label]
    self.debug = debug

    self._end_delay = end_delay
    self._spam = spam
    self._safe_start = safe_start
    self._safe_started = False

    # The path is an iterable object
    self._path = cycle(path) if repeat else iter(path)

    # More attributes
    self._ended_no_raise = False
    self._last_cmd = None
    self._last_id = None
    self._last_t = None
    self._current_path = None
    self._path_id = None

  def prepare(self) -> None:
    """Checks the validity of the provided path."""

    # Checking the validity of the path
    try:
      self._check_path_validity(iter(deepcopy(self._path)))
    except (Exception,):
      self.log(logging.ERROR, "Exception raised while parsing the generator "
                              "path !")
      raise

  def begin(self) -> None:
    """Initializes the first path and runs a :meth:`loop`, that may be
    blocking."""

    self._update_path()

  def loop(self) -> None:
    """First reads data from upstream blocks, then gets the next command to
    send, and finally sends it to downstream blocks.

    It also manages the transitions between the paths.
    """

    # Case when the Generator shouldn't raise CrappyStop after it ended
    if self._ended_no_raise:
      return

    # If self start requested, do nothing until the first values are received
    if self._safe_start and not self._safe_started:
      if self.data_available():
        self._safe_started = True
        self.log(logging.INFO, "First data received, starting safely")
      else:
        self.log(logging.DEBUG, "Waiting for first data to arrive for "
                                "starting safely")
        return

    # Getting the data from upstream blocks
    data = self.recv_all_data()
    try:
      # Getting the next command to send
      self._last_t = time()
      cmd = self._current_path.get_cmd(data)
      self.log(logging.DEBUG, f"Next command: {cmd}")
    except StopIteration:
      try:
        # Switching to the next path if we reached the end of one
        self._update_path()
        self.loop()
        return
      except GeneratorNoStop:
        self.log(logging.WARNING, f"Generator path exhausted, staying idle "
                                  f"until the script ends")
        # Case when the Generator shouldn't raise CrappyStop after it ended
        self._ended_no_raise = True
        return

    # The command is sent if it's different from the previous, or if we
    # switched to the next path, or if spam is True
    if cmd != self._last_cmd or self._last_id != self._path_id or self._spam:
      self._last_cmd = cmd
      self._last_id = self._path_id
      # Actually sending the command
      self.send([self._last_t - self.t0, cmd, self._path_id])

  def _update_path(self) -> None:
    """Gets the next path from the list of paths and instantiates it.

    Also manages the case when the last path of the list was reached.
    """

    try:
      # Getting the next path from the list of paths
      next_path_dict = deepcopy(next(self._path))
      # Updating the path index
      if self._path_id is not None:
        self._path_id += 1
      else:
        self._path_id = 0

    # Raised when the list of paths is exhausted
    except StopIteration:
      # First option, stopping the program after a delay
      if self._end_delay is not None:
        sleep(self._end_delay)
        raise GeneratorStop
      # Second option, not stopping the program and looping forever
      else:
        raise GeneratorNoStop

    self.log(logging.INFO, f"Next generator path (id: {self._path_id}): "
                           f"{next_path_dict['type']}")

    # Instantiating the next generator path object
    path_name = next_path_dict.pop('type')
    if path_name in paths_dict:
      path_type = paths_dict[path_name]
    else:
      raise ValueError(f"No Generator path called {path_name}! The available"
                       f" generator paths are : {tuple(paths_dict.keys())}")
    self._current_path = path_type(
      _last_time=self._last_t if self._last_t is not None else self.t0,
      _last_cmd=self._last_cmd,
      **next_path_dict)

  @staticmethod
  def _check_path_validity(path: Iterator[Dict[str, Any]]) -> None:
    """Simply instantiates all the paths in a row to check no error is
    raised."""

    for i, next_dict in enumerate(path):
      next_dict = deepcopy(next_dict)
      path_name = next_dict.pop('type')
      if path_name in paths_dict:
        path_type = paths_dict[path_name]
      else:
        raise ValueError(f"No Generator path called {path_name}! The available"
                         f" generator paths are : {tuple(paths_dict.keys())}")
      path_type(_last_time=0, _last_cmd=None if i == 0 else 0, **next_dict)
