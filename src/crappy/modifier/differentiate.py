# coding: utf-8

from typing import Optional, Dict, Any
import logging

from .meta_modifier import Modifier


class Diff(Modifier):
  """This Modifier calculates the time derivative of a given label and adds the
  derivative to the returned data."""

  def __init__(self,
               label: str,
               time_label: str = 't(s)',
               out_label: Optional[str] = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The label whose time derivative to compute.
      time_label: The label carrying the time information.
      out_label: The label carrying the calculated derivative. If not given,
        defaults to ``'d_<label>'``.
    """

    super().__init__()
    self._label = label
    self._time_label = time_label
    self._out_label = out_label if out_label is not None else f'd_{label}'

    self._last_t = None
    self._last_val = None

  def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the data from the upstream Block, updates the derivative value,
    appends it to the data and returns the data."""

    self.log(logging.DEBUG, f"Received {data}")

    # For the first received data, storing it and returning 0
    if self._last_t is None or self._last_val is None:
      self._last_t = data[self._time_label]
      self._last_val = data[self._label]
      data[self._out_label] = 0
      return data

    # Updating the differentiation value with the latest received values
    t = data[self._time_label]
    val = data[self._label]
    diff = (val - self._last_val) / (t - self._last_t)
    # Updating the stored data
    self._last_t = t
    self._last_val = val

    # Returning the updated data
    data[self._out_label] = diff
    self.log(logging.DEBUG, f"Sending {data}")
    return data
