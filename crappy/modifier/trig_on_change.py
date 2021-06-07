# coding: utf-8

from typing import Union
from .modifier import Modifier


class Trig_on_change(Modifier):
  """Can be used to trig an event when the value of a given label changes."""

  def __init__(self, name: str) -> None:
    """Sets the instance attributes.

    Args:
      name (:obj:`str`): The name of the label to monitor.
    """

    self.name = name

  def evaluate(self, data: dict) -> Union[dict, None]:
    if not hasattr(self, 'last'):
      self.last = data[self.name]
      return data
    if data[self.name] == self.last:
      return None
    self.last = data[self.name]
    return data
