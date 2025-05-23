# coding: utf-8

from typing import Optional
import logging

from .meta_modifier import Modifier, T


class TrigOnChange(Modifier):
  """Modifier passing the data to the downstream Block only when the value of
  a given label changes.

  It also transmits the first received data. Can be used to trigger a Block
  upon change of a label value.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Trig_on_change* to *TrigOnChange*
  """

  def __init__(self, label: str) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The name of the label to monitor.
    
        .. versionchanged:: 1.5.10 renamed from *name* to *label*
    """

    super().__init__()
    self._label = label
    self._last = None

  def __call__(self, data: dict[str, T]) -> Optional[dict[str, T]]:
    """Compares the received value with the last sent one, and if they're
    different sends the received data and stores the latest value.
    
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

    # Storing the first received value and returning the data
    if self._last is None:
      self._last = data[self._label]
      self.log(logging.DEBUG, f"Sending {data}")
      return data

    # Returning the data if the label value is different from the stored value
    if data[self._label] != self._last:
      self._last = data[self._label]
      self.log(logging.DEBUG, f"Sending {data}")
      return data

    self.log(logging.DEBUG, "Not returning any data")
