# coding: utf-8

from typing import Optional, Dict, Any
import logging
from multiprocessing import current_process

from .meta_modifier import MetaModifier


class Modifier(metaclass=MetaModifier):
  """The base class for all modifier classes, simply allowing to keep track of
  them.

  The Modifiers allow altering data from an input Block before it gets sent to
  an output Block.
  """

  def __init__(self, *_, **__) -> None:
    """Sets the logger attribute."""

    self._logger: Optional[logging.Logger] = None

  def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The main method altering the inout data and returning the altered data.

    Args:
      data: The data from the input Block, as a :obj:`dict`.

    Returns:
      Data to send to the output Block, as a :obj:`dict`. It is also fine for
      this method not to return anything, in which case no message is
      transmitted to the output Block.
    """

    self.log(logging.DEBUG, f"Received {data}")
    self.log(logging.WARNING, "The __call__ method is not defined, not "
                              "altering the data !")
    self.log(logging.DEBUG, f"Sending {data}")
    return data

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Modifiers.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
