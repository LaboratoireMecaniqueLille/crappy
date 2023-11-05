# coding: utf-8

from .modifier import Modifier
from typing import Optional, Tuple, Dict, Any, Union, List
from warnings import warn


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

    warn("The evaluate method will be renamed to __call__ in version 2.0.0",
         FutureWarning)

    if data[self._label] in self._values:
      return data
