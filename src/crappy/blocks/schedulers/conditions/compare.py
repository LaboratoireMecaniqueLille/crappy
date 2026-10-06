# coding: utf-8

from collections.abc import Callable, Container
from itertools import product
from numbers import Real
from typing import Any, Literal
import logging
import operator

from .condition import Condition


class Compare(Condition):
  """Compares received values with a given constant or another input label.

  With a second label, ``'any'`` and ``'all'`` evaluate the Cartesian product
  of the two labels' value lists. The ``'in'`` and ``'not in'`` operations
  compare a received value with a constant container only.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               label_1: str,
               operation: Literal['<', '>', '<=', '>=', '!=', '==', 'in',
                                  'not in'],
               value: Any | None = None,
               label_2: str | None = None,
               mode: Literal['any', 'all', 'last'] = 'any') -> None:
    """Sets the operands, comparison operator and value-selection mode.

    Args:
      label_1: The non-empty label containing the left operand.
      operation: One of ``'<'``, ``'>'``, ``'<='``, ``'>='``, ``'!='``,
        ``'=='``, ``'in'``, or ``'not in'``.
      value: A constant right operand. Provide either this argument or
        ``label_2``. Membership operations require a container here.
      label_2: The non-empty label containing the right operand when comparing
        two input labels. Cannot be used with membership operations.
      mode: ``'any'`` succeeds if any value-to-value comparison is true,
        ``'all'`` requires every value-to-value comparison to be true,
        ``'last'`` compares only the latest received value for each label.
    """

    super().__init__()

    match label_1:
      case str() if label_1.strip():
        self._label_1: str = label_1
      case str():
        raise ValueError("The provided label_1 must be a non-empty string")
      case _:
        raise TypeError("The provided label_1 must be a non-empty string")

    match operation:
      case str() if operation in ('<', '>', '<=', '>=', '!=', '==', 'in',
                                  'not in'):
        pass
      case str():
        raise ValueError("operation must be either '<', '>', '<=', '>=', "
                         "'!=', '==', 'in', or 'not in'")
      case _:
        raise TypeError("operation must be either '<', '>', '<=', '>=', '!=', "
                        "'==', 'in', or 'not in'")

    match label_2:
      case None:
        self._label_2: str | None = label_2
      case str() if label_2.strip():
        self._label_2: str | None = label_2
      case str():
        raise ValueError("The provided label_2 must be a non-empty string")
      case _:
        raise TypeError("The provided label_2 must be a non-empty string")

    match mode:
      case str() if mode in ('any', 'all', 'last'):
        self._mode: Literal['any', 'all', 'last'] = mode
      case str():
        raise ValueError("mode must be either 'any', 'all', or 'last'")
      case _:
        raise TypeError("mode must be either 'any', 'all', or 'last'")

    # Check the possible combinations
    if value is None and label_2 is None:
      raise ValueError("Both value and label_2 are None, one of them must be "
                       "provided")
    if value is not None and label_2 is not None:
      raise ValueError("Both value and label_2 are provided, set one only")
    if value is None and operation in ('in', 'not in'):
      raise ValueError("The operations 'in' and 'not in' only support "
                       "comparison with a value, not with a label")

    match operation:
      case '<':
        if value is not None and not isinstance(value, Real):
          raise ValueError("When comparing against a value with operation "
                           "'<', the value must be a real number")
        self._operator: Callable[[Any, Any], bool] = operator.lt
      case '>':
        if value is not None and not isinstance(value, Real):
          raise ValueError("When comparing against a value with operation "
                           "'>', the value must be a real number")
        self._operator: Callable[[Any, Any], bool] = operator.gt
      case '<=':
        if value is not None and not isinstance(value, Real):
          raise ValueError("When comparing against a value with operation "
                           "'<=', the value must be a real number")
        self._operator: Callable[[Any, Any], bool] = operator.le
      case '>=':
        if value is not None and not isinstance(value, Real):
          raise ValueError("When comparing against a value with operation "
                           "'>=', the value must be a real number")
        self._operator: Callable[[Any, Any], bool] = operator.ge
      case '!=':
        if value is not None and not isinstance(value, (Real, str)):
          raise ValueError("When comparing against a value with operation "
                           "'!=', the value must be a real number or a string")
        self._operator: Callable[[Any, Any], bool] = operator.ne
      case '==':
        if value is not None and not isinstance(value, (Real, str)):
          raise ValueError("When comparing against a value with operation "
                           "'==', the value must be a real number or a string")
        self._operator: Callable[[Any, Any], bool] = operator.eq
      case 'in':
        if not isinstance(value, Container):
          raise ValueError("With operation 'in', the provided value must be a "
                           "container")
        self._operator = self.contains
      case 'not in':
        if not isinstance(value, Container):
          raise ValueError("With operation 'not in', the provided value must "
                           "be a container")
        self._operator = self.not_contains

    self._value: Any = value

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Returns the comparison result, or :obj:`False` if a label is absent
    or has no values."""

    # First check that the condition can be evaluated
    if self._label_1 not in data or not data[self._label_1]:
      self.log(logging.DEBUG, f"label {self._label_1} missing from the input "
                              f"data, cannot check the condition")
      return False
    if self._label_2 is not None and (self._label_2 not in data or
                                      not data[self._label_2]):
      self.log(logging.DEBUG, f"label {self._label_2} missing from the input "
                              f"data, cannot check the condition")
      return False

    if self._mode == 'any':
      if self._label_2 is not None:
        return any(self._operator(val_1, val_2) for val_1, val_2
                   in product(data[self._label_1], data[self._label_2]))
      else:
        return any(self._operator(val_1, self._value)
                   for val_1 in data[self._label_1])
    elif self._mode == 'all':
      if self._label_2 is not None:
        return all(self._operator(val_1, val_2) for val_1, val_2
                   in product(data[self._label_1], data[self._label_2]))
      else:
        return all(self._operator(val_1, self._value)
                   for val_1 in data[self._label_1])
    else:
      if self._label_2 is not None:
        return self._operator(data[self._label_1][-1], data[self._label_2][-1])
      else:
        return self._operator(data[self._label_1][-1], self._value)

  @staticmethod
  def contains(a, b: Container) -> bool:
    """Convenience method for inverting the operands of
    :func:`operator.contains`."""

    return operator.contains(b, a)

  @staticmethod
  def not_contains(a, b: Container) -> bool:
    """Convenience method for inverting the operands of
    :func:`operator.contains`."""

    return not operator.contains(b, a)
