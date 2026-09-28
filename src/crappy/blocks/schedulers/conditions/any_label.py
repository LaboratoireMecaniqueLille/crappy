# coding: utf-8

from typing import Any, Literal
import logging

from .condition import Condition


class AnyLabel(Condition):
  """Applies a truthiness rule to any named input label.

  Missing labels and labels with empty value lists are ignored. The ``mode``
  determines which values are checked within each remaining label's list.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               *labels: str,
               mode: Literal['any', 'all', 'last'] = 'any') -> None:
    """Sets the labels and the rule applied to each present label.

    Args:
      *labels: One or more non-empty labels to check.
      mode: ``'any'`` accepts a label with at least one truthy value,
        ``'all'`` accepts a label whose values are all truthy, ``'last'``
        accepts a label whose latest value is truthy.
    """

    super().__init__()

    match labels:
      case ():
        raise ValueError("At least one label must be provided")
      case (*labels,) if (all(isinstance(label, str) for label in labels) and
                          all(label.strip() for label in labels)):
        self._labels: tuple[str, ...] = tuple(labels)
      case (*_,):
        raise ValueError("labels must be provided as non-empty strings")
      case _:
        raise TypeError("labels must be provided as non-empty strings")

    match mode:
      case str() if mode in ('any', 'all', 'last'):
        self._mode: Literal['any', 'all', 'last'] = mode
      case str():
        raise ValueError("mode must be either 'any', 'all', or 'last'")
      case _:
        raise TypeError("mode must be either 'any', 'all', or 'last'")

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Returns whether any present label meets the truthiness rule."""

    # A missing value is not blocking but is reported
    if any(label not in data for label in self._labels):
      missing = [label for label in self._labels if label not in data]
      self.log(logging.DEBUG, f"Values missing for label(s) "
                              f"{', '.join(missing)}, still proceeding")
    # An empty list doesn't allow to proceed
    if any(not data[label] for label in self._labels if label in data):
      missing = [label for label in data if not data[label]]
      self.log(logging.DEBUG, f"Empty lists received for label(s) "
                              f"{', '.join(missing)}, ignoring them")
      data = {label: values for label, values in data.items() if data[label]}

    if self._mode == 'any':
      return any(any(data[label]) for label in self._labels if label in data)
    elif self._mode == 'all':
      return any(all(data[label]) for label in self._labels if label in data)
    else:
      return any(data[label][-1] for label in self._labels if label in data)
