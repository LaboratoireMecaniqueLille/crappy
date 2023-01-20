# coding: utf-8

from typing import Optional, Dict, Any
import logging
from multiprocessing import current_process

from .meta_modifier import MetaModifier


class Modifier(metaclass=MetaModifier):
  """The base class for all modifier classes, simply allowing to keep track of
  them."""

  def __init__(self, *_, **__) -> None:
    """"""

    self._logger: Optional[logging.Logger] = None

  def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """"""

    self.log(logging.DEBUG, f"Received {data}")
    self.log(logging.WARNING, "The evaluate method is not defined, not "
                              "altering the data !")
    self.log(logging.DEBUG, f"Sending {data}")
    return data

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
