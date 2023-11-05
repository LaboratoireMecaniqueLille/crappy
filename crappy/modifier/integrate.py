# coding: utf-8

from .modifier import Modifier
from typing import Optional, Dict, Any
from warnings import warn


class Integrate(Modifier):
  """This modifier integrates the data of a label over time and adds the
  integration value to the returned data."""

  def __init__(self,
               label: str,
               time_label: str = 't(s)',
               out_label: Optional[str] = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The label whose data to integrate over time.
      time_label: The label carrying the time information.
      out_label: The label carrying the integration value. If not given,
        defaults to ``'i_<label>'``.
    """

    super().__init__()
    self._label = label
    self._time_label = time_label
    self._out_label = out_label if out_label is not None else f'i_{label}'

    self._last_t = None
    self._last_val = None
    self._integration = 0

  def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the data from the upstream block, updates the integration value and
    returns it."""

    warn("The evaluate method will be renamed to __call__ in version 2.0.0",
         FutureWarning)

    # For the first received data, storing it and returning 0
    if self._last_t is None or self._last_val is None:
      self._last_t = data[self._time_label]
      self._last_val = data[self._label]
      data[self._out_label] = self._integration
      return data

    # Updating the integrated value with the latest received values
    t = data[self._time_label]
    val = data[self._label]
    self._integration += (t - self._last_t) * (val + self._last_val) / 2
    # Updating the stored data
    self._last_t = t
    self._last_val = val

    # Returning the updated data
    data[self._out_label] = self._integration
    return data
