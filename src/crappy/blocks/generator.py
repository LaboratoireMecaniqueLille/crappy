# coding: utf-8

from time import time, sleep
from typing import Any, Optional
from collections.abc import Iterator, Iterable
from itertools import cycle
from copy import deepcopy
import logging

from .meta_block import Block
from .generator_path.meta_path import paths_dict, Path
from .._global import GeneratorStop


class GeneratorNoStop(Exception):
  """A custom exception for handling the case when the Generator should not
  raise a :exc:`CrappyStop` exception when it terminates."""


class Generator(Block):
  """This Block generates a signal following a user-defined assembly of
  :class:`~crappy.blocks.generator_path.meta_path.Path`.
  
  The generated signal is just a waveform that can serve any purpose. It can
  for example be used for driving a :class:`~crappy.blocks.Machine` Block, or 
  for triggering a :class:`~crappy.blocks.Camera` Block.

  One Generator Block can only generate one signal. Use multiple Blocks if
  several signals are needed. Note that the default behavior of a Generator
  is to stop the entire script when it reaches the end of all the Paths.

  This Block can also accept inputs from other Blocks, as these inputs may be
  used by a :class:`~crappy.blocks.generator_path.meta_path.Path`. The most
  common use of this feature is to have the stop condition of a Path depend on
  the received values of a label.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               path: Iterable[dict[str, Any]],
               freq: Optional[float] = 200,
               cmd_label: str = 'cmd',
               path_index_label: str = 'index',
               repeat: bool = False,
               spam: bool = False,
               display_freq: bool = False,
               end_delay: Optional[float] = 2,
               safe_start: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      path: An iterable (like a :obj:`list` or a :obj:`tuple`) of :obj:`dict`,
        each dict providing the parameters to generate a 
        :class:`~crappy.blocks.generator_path.meta_path.Path`. The Paths are
        generated in the order in which they are given, and the stop condition
        of each Path is used for determining when to switch to the next one.
        The ``'type'`` key of each :obj:`dict` gives the name of the Path to
        use, and all the other keys correspond to the arguments to give to
        this Path. Refer to the documentation of the chosen Paths to know which
        keys to provide.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      cmd_label: The label of the signal sent to the downstream Blocks.
      path_index_label: In addition to the ``cmd_label``, this label holds the
        index of the current
        :class:`~crappy.blocks.generator_path.meta_path.Path`. Useful to
        trigger a Block when the current Path changes, as the output value
        might not necessarily change.
      repeat: If :obj:`True`, the ``path`` will loop forever instead of
        stopping when it reaches the last Path.
      spam: If :obj:`True`, the signal value will be sent on each loop. Else,
        it will only be sent if it is different from the previous or if the
        Block switched to the next Path.
      display_freq: if :obj:`True`, displays the looping frequency of the 
        Block.
        
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      end_delay: When all the Paths are exhausted, waits this many seconds
        before stopping the entire script. Can be set to :obj:`None`,
        in which case the Generator won't stop the program when finishing.
      safe_start: Ensures the first Path waits for at least one data point from
        upstream Blocks before sending the first value of the signal.
        Otherwise, the first value might be sent without checking the
        associated condition if its depends on labels from other Blocks.
        
        .. versionadded:: 1.5.10
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    super().__init__()

    # Instantiating a few attributes based on the arguments
    self.niceness = -5
    self.freq = freq
    self.display_freq = display_freq
    self.labels = ['t(s)', cmd_label, path_index_label]
    self.debug = debug

    self._end_delay = end_delay
    self._spam = spam
    self._safe_start = safe_start
    self._safe_started = False

    # The path is an iterable object
    path = list(path)
    self._path = cycle(path) if repeat else iter(path)

    # More attributes
    self._ended_no_raise = False
    self._last_cmd = None
    self._last_id = None
    self._last_t = None
    self._current_path = None
    self._path_id = None

    # Checking the validity of the path
    self._check_path_validity(iter(deepcopy(iter(path))))

  def begin(self) -> None:
    """Initializes the first
    :class:`~crappy.blocks.generator_path.meta_path.Path`."""

    self._update_path()

  def loop(self) -> None:
    """First reads data from upstream Blocks, then gets the next command to
    send, and finally sends it to downstream Blocks.

    It also manages the transitions between the
    :class:`~crappy.blocks.generator_path.meta_path.Path`.
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
      self.log(logging.DEBUG, f"Returned command: {cmd}")
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

    # Not sending if no command was output by the get_cmd method
    if cmd is None:
      return

    # The command is sent if it's different from the previous, or if we
    # switched to the next path, or if spam is True
    if cmd != self._last_cmd or self._last_id != self._path_id or self._spam:
      self._last_cmd = cmd
      self._last_id = self._path_id
      # Actually sending the command
      self.send([self._last_t - self.t0, cmd, self._path_id])

  def _update_path(self) -> None:
    """Gets the next Path from the list of Paths and instantiates it.

    Also manages the case when the last Path of the list was reached.
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
    self._check_path_exists(path_name)
    path_type = paths_dict[path_name]
    Path.t0 = self._last_t if self._last_t is not None else self.t0
    Path.last_cmd = self._last_cmd
    self._current_path = path_type(**next_path_dict)

  def _check_path_validity(self, path: Iterator[dict[str, Any]]) -> None:
    """Simply instantiates all the Paths in a row to check no error is
    raised."""

    for i, next_dict in enumerate(path):
      next_dict = deepcopy(next_dict)
      path_name = next_dict.pop('type')
      self._check_path_exists(path_name)
      Path.t0 = 0
      Path.last_cmd = None if i == 0 else 0
      path_type = paths_dict[path_name]
      path_type(**next_dict)

  @staticmethod
  def _check_path_exists(name: str) -> None:
    """Checks that the provided Generator Path is a valid one, and raises an
    error if not."""

    if name not in paths_dict:
      possible = ', '.join(sorted(paths_dict.keys()))
      raise ValueError(f"Unknown Generator path type : {name} ! "
                       f"The possible types are : {possible}")
