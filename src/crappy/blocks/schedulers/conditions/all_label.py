# coding: utf-8

from typing import Any, Literal
import logging

from .condition import Condition


class AllLabel(Condition):
  """Applies a truthiness rule to every named input label.

  The ``mode`` determines which values are checked within each label's list.
  All named labels must be present and have at least one value for this
  condition to be true.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               *labels: str,
               mode: Literal['any', 'all', 'last'] = 'any') -> None:
    """Sets the labels and the rule applied to each label's values.

    Args:
      *labels: One or more non-empty labels to check.
      mode: ``'any'`` requires at least one truthy value per label,
        ``'all'`` requires every value per label to be truthy, ``'last'``
        checks the latest value per label.
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
    """Returns whether all labels meet the configured truthiness rule."""

    # A missing value is interpreted as False
    if (any(label not in data for label in self._labels) or
        any(not data[label] for label in self._labels)):
      missing = [label for label in self._labels if label not in data or
                 not data[label]]
      self.log(logging.DEBUG, f"Cannot make a decision, values missing for "
                              f"label(s) {', '.join(missing)}")
      return False

    if self._mode == 'any':
      return all(any(data[label]) for label in self._labels)
    elif self._mode == 'all':
      return all(all(data[label]) for label in self._labels)
    else:
      return all(data[label][-1] for label in self._labels)
