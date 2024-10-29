# coding: utf-8

from time import time
from typing import Optional
import numpy as np

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from psutil import virtual_memory
except (ModuleNotFoundError, ImportError):
  virtual_memory = OptionalModule("psutil")


class FakeInOut(InOut):
  """This class is a demonstration InOut object that does not require any
  hardware to run.

  It can read and/or modify (to a certain extent) the memory usage of the
  computer.
  
  .. versionadded:: 1.5.5
  .. versionchanged:: 2.0.0 renamed from *Fake_inout* to *FakeInOut*
  """

  def __init__(self) -> None:
    """Initializes the parent class."""

    self._buf: Optional[list] = None

    super().__init__()

  def open(self) -> None:
    """Creates the buffer allowing to modify the memory usage."""

    self._buf = list()

  def set_cmd(self, *cmd: float) -> None:
    """Modifies the memory usage of the computer.

    If the command is higher than the current, adds big lists to the buffer in
    order to use more memory.
    If the command is lower than the current, empties the buffer.

    Args:
      *cmd: The target memory usage to set, in percent as a :obj:`float`.
    """

    if not isinstance(cmd[0], float) and not isinstance(cmd[0], int):
      raise TypeError("Not the right command type for the FakeInOut !")
    if not 0 <= cmd[0] <= 100:
      raise ValueError("Command should be a percentage of memory usage !")

    if cmd[0] > virtual_memory().percent:
      self._buf.append([0] * 1024*1024)
    elif cmd[0] < virtual_memory().percent:
      try:
        # If there's nothing to delete, abort
        del self._buf[-1]
      except IndexError:
        return

  def get_data(self) -> list[float]:
    """Just returns the timestamp and the current memory usage."""

    return [time(), virtual_memory().percent]

  def start_stream(self) -> None:
    """Defining this method to avoid getting warnings in the logs.
    
    .. versionadded:: 2.0.0
    """

    ...

  def stop_stream(self) -> None:
    """Defining this method to avoid getting warnings in the logs.

    .. versionadded:: 2.0.0
    """

    ...

  def get_stream(self) -> tuple[np.ndarray, np.ndarray]:
    """This method calls 10 times the :meth:`get_data` method and returns the
    10 values at once in the streamer format.

    It is just a demo for showcasing the use of the streamer mode.
    
    .. versionadded:: 2.0.0
    """

    values = np.array([self.get_data() for _ in range(10)])
    t, ret = np.split(values, 2, axis=1)
    return np.squeeze(t), ret

  def close(self) -> None:
    """Deletes the buffer."""

    del self._buf
