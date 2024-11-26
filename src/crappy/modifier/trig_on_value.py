# coding: utf-8

from typing import Optional, Any, Union
from collections.abc import Iterable
import logging

from .meta_modifier import Modifier, T


class TrigOnValue(Modifier):
  """Modifier passing the data to the downstream Block only if the value
  carried by a given label matches a given set of accepted values.

  Mostly useful to trigger Blocks in predefined situations.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Trig_on_value* to *TrigOnValue*
  """

  def __init__(self,
               label: str,
               values: Union[Any, Iterable[Any]]) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The name of the label to monitor.

        .. versionchanged:: 1.5.10 renamed from *name* to *label*
      values: The values of ``label`` for which the data will be transmitted.
        Can be a single value, or an iterable of values (like a :obj:`list` or
        a :obj:`tuple`).
    """

    super().__init__()

    self._label = label

    if isinstance(values, str):
      values = (values,)
    else:
      try:
        iter(values)
      except TypeError:
        values = (values,)

    self._values = tuple(values)

  def __call__(self, data: dict[str, T]) -> Optional[dict[str, T]]:
    """Checks if the value of ``label`` is in the predefined set of accepted
    values, and if so transmits the data.
    
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

    if data[self._label] in self._values:
      self.log(logging.DEBUG, f"Sending {data}")
      return data

    self.log(logging.DEBUG, "Not returning any data")
