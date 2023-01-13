# coding: utf-8

from typing import Optional
import logging
from multiprocessing import current_process

from .meta_modifier import MetaModifier


class Modifier(metaclass=MetaModifier):
  """The base class for all modifier classes, simply allowing to keep track of
  them."""

  def __init__(self, *_, **__) -> None:
    """"""

    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
