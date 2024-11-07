# coding: utf-8

from typing import Optional, TypeVar
import logging
from multiprocessing import current_process

from .meta_modifier import MetaModifier

T = TypeVar('T')
"""Generic type representing data handled by Modifiers."""


class Modifier(metaclass=MetaModifier):
  """The base class for all Modifier classes, simply allowing to keep track of
  them.

  The Modifiers allow altering data from an input :class:`~crappy.blocks.Block`
  before it gets sent to an output Block. Each Modifier is associated to a
  :class:`~crappy.links.Link` linking the two Blocks. It is passed as an
  argument of the :meth:`~crappy.link` method instantiating the Link.

  It is preferable for every Modifier to be a child of this class, although
  that is not mandatory. A Modifier only needs to be a callable, i.e. a class
  defining the :meth:`~crappy.modifier.Modifier.__call__` method or a function.

  .. versionadded:: 1.4.0
  """

  def __init__(self, *_, **__) -> None:
    """Sets the logger attribute.

    .. versionchanged:: 2.0.0 now accepts args and kwargs
    """

    self._logger: Optional[logging.Logger] = None

  def __call__(self, data: dict[str, T]) -> Optional[dict[str, T]]:
    """The main method altering the inout data and returning the altered data.

    It should take a :obj:`dict` as its only argument, and return another
    :obj:`dict`. Both dicts should have their keys as :obj:`str`, representing
    the labels. Their values constitute the data flowing through the
    :class:`~crappy.links.Link`.

    Args:
      data: The data from the input :class:`~crappy.blocks.Block`, as a
        :obj:`dict`.

    Returns:
      Data to send to the output :class:`~crappy.blocks.Block`, as a
      :obj:`dict`. It is also fine for this method to return :obj:`None`, in
      which case no message is transmitted to the output Block.

    .. versionadded:: 2.0.0
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
    
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
