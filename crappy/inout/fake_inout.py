# coding: utf-8

from time import time
from .inout import InOut
from .._global import OptionalModule

try:
  from psutil import virtual_memory
except (ModuleNotFoundError, ImportError):
  psutil = OptionalModule("psutil")


class Fake_inout(InOut):
  """A class demonstrating the usage of an inout abject without requiring any
  hardware.

  It can read and/or modify the current memory usage on the computer.
  """

  def __init__(self) -> None:
    """Not much to do here."""

    super().__init__()

  def open(self) -> None:
    """Creates the buffer allowing to modify the memory usage."""

    self._buf = list()

  def set_cmd(self, *cmd: float) -> None:
    """Modifies the computer memory usage.

    If the command is lower than the current, empties the buffer.
    If the command is higher than the current, adds big lists to the buffer in
    order to use more memory.

    Args:
      *cmd (:obj:`float`): The target memory usage to set.
    """

    if not isinstance(cmd[0], float) and not isinstance(cmd[0], int):
      raise TypeError("Not the right command type for the Fake_inout !")
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

  @staticmethod
  def get_data() -> list:
    """Just returns time and the current memory usage."""

    return [time(), virtual_memory().percent]

  def close(self) -> None:
    """Deletes the buffer."""

    del self._buf
