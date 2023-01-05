# coding: utf-8

from typing import Optional, Tuple, Dict, Any, Union, List
import logging

from .modifier import Modifier


class Trig_on_value(Modifier):
  """Modifier passing the data to the downstream only if the value carried by a
  given label matches a given set of accepted values.

  Mostly useful to trig blocks.
  """

  def __init__(self,
               label: str,
               values: Union[Any, Tuple[Any, ...], List[Any]]) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The name of the label to monitor.
      values: The values of ``label`` for which the data will be transmitted.
        Can be a single value, a :obj:`list` or a :obj:`tuple`.
    """

    super().__init__()

    self._label = label
    if isinstance(values, list) or isinstance(values, tuple):
      self._values = values
    else:
      self._values = (values,)

  def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Checks if the value of ``label`` is in the predefined set of accepted
    values, and if so transmits the data."""

    self.log(logging.DEBUG, f"Received {data}")

    if data[self._label] in self._values:
      self.log(logging.DEBUG, f"Sending {data}")
      return data

    self.log(logging.DEBUG, "Not returning any data")
