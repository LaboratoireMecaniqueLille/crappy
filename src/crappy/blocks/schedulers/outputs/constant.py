# coding: utf-8

from typing import Any

from .output import Output


class Constant(Output):
  """Outputs the same value for one label on every call.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               label: str,
               value: Any) -> None:
    """Sets the output label and its constant value.

    Args:
      label: The non-empty label under which to send the value.
      value: The value to output.
    """

    super().__init__()

    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")

    self._value: Any = value

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, Any]:
    """Returns the configured value under its output label."""

    return {self._label: self._value}
