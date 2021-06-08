# coding: utf-8

from .modifier import Modifier
from typing import Union


class Trig_on_value(Modifier):
  """Can be used to send data (an empty :obj:`dict`) when the input reached a
  given value.

  Note:
    The modifier will trig if `data[name]` is in ``values``.
  """

  def __init__(self, name: str, values: list) -> None:
    """Sets the instance attributes.

    Args:
      name (:obj:`str`): The name of the label to monitor.
      values (:obj:`list`): A list containing the possible values to send the
        signal.
    """

    self.name = name
    self.values = values if isinstance(values, list) else [values]

  def evaluate(self, data: dict) -> Union[dict, None]:
    if data[self.name] in self.values:
      return data
