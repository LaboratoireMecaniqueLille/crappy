# coding: utf-8

from typing import Any
from multiprocessing import current_process
from abc import ABC, abstractmethod
import logging


class Condition(ABC):
  """Base class for conditions that trigger Scheduler State transitions.

  Subclasses implement :meth:`__call__` to decide from elapsed State time
  and received data whether a transition should occur. The optional
  :meth:`reset` hook is called when the Scheduler enters a State containing
  the Condition.

  .. versionadded:: 2.1.0
  """

  def __init__(self) -> None:
    """Initializes the logger used by the Condition."""

    self._logger: logging.Logger | None = None

  @abstractmethod
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Checks whether the State should take this transition.

    Args:
      dt: The time elapsed since the current State was entered, in seconds.
      data: Values received from upstream Blocks, grouped by label.

    Returns:
      :obj:`True` when the transition condition is met, otherwise
      :obj:`False`.
    """

    ...

  def reset(self) -> None:
    """Hook for clearing per-State history. The default does nothing."""

    ...

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Condition.

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
