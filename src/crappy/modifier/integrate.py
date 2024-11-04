# coding: utf-8

from typing import Optional
import logging

from .meta_modifier import Modifier, T


class Integrate(Modifier):
  """This Modifier integrates the data of a label over time and adds the
  integration value to the returned data.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               label: str,
               time_label: str = 't(s)',
               out_label: Optional[str] = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      label: The label whose data to integrate over time.
      time_label: The label carrying the time information.

        .. versionchanged:: 1.5.10 renamed from *time* to *time_label*
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

  def __call__(self, data: dict[str, T]) -> dict[str, T]:
    """Gets the data from the upstream Block, updates the integration value,
    adds it to the data and returns the data.
    
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

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
    self.log(logging.DEBUG, f"Sending {data}")
    return data
