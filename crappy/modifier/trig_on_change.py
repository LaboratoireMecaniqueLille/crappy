# coding: utf-8

from typing import Optional, Dict, Any
import logging

from .modifier import Modifier


class Trig_on_change(Modifier):
  """Modifier passing the data to the downstream block only when the value of
  a given label changes.

  It also transmits the first received data. Can be used to trig a block upon
  change of a label value.
  """

  def __init__(self, label: str) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The name of the label to monitor.
    """

    super().__init__()
    self._label = label
    self._last = None

  def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compares the received value with the last sent one, and if they're
    different sends the received data and stores the latest value."""

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
