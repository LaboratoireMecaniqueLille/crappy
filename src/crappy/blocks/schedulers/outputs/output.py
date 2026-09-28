# coding: utf-8

from typing import Any
from multiprocessing import current_process
from abc import ABC, abstractmethod
import logging


class Output(ABC):
  """Base class for output functions used by a
  :class:`~crappy.blocks.Scheduler` State.

  Subclasses implement :meth:`__call__` to generate values from the elapsed
  time in the current State and data received from upstream Blocks. The
  optional :meth:`reset` hook is called when the Scheduler enters a State
  containing the Output.

  .. versionadded:: 2.1.0
  """

  def __init__(self) -> None:
    """Initializes the logger used by the Output."""

    self._logger: logging.Logger | None = None

  @abstractmethod
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, Any] | None:
    """Generates values for one or more Scheduler output labels.

    Args:
      dt: The time elapsed since the current State was entered, in seconds.
      data: Values received from upstream Blocks, grouped by label.

    Returns:
      A dictionary mapping output labels to values, or :obj:`None` when no
      value is generated on this call.
    """

    ...

  def reset(self) -> None:
    """Hook for clearing per-State history. The default does nothing."""

    ...

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Output.

    Also instantiates the :obj:`~logging.Logger` when logging the first
    message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
