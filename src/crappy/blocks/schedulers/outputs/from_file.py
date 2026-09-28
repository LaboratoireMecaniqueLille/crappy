# coding: utf-8

from typing import Any
from pathlib import Path
import logging
import numpy as np

from .output import Output


class FromFile(Output):
  """Outputs a time-varying value interpolated from a two-column file.

  The first column contains times in seconds since the State was entered,
  the second contains output values. The data is loaded when this Output is
  constructed and linearly interpolated on each call.

  .. versionadded:: 2.1.0
  """
  
  def __init__(self, 
               label: str,
               file_name: Path | str,
               delimiter: str = ',',
               repeat_last: bool = False) -> None:
    """Loads the time and value columns from a text file.

    Args:
      label: The non-empty label under which to send the value.
      file_name: Path to a file containing exactly two numeric columns. Times
        in the first column must be strictly increasing, and both columns must
        contain only finite values.
      delimiter: The separator used between the file's columns.
      repeat_last: If :obj:`True`, keeps outputting the final value after the
        last timestamp. Otherwise, returns :obj:`None` after that time.
    """
    
    super().__init__()
    
    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")

    match file_name:
      case Path() if file_name.name:
        self._file_name: Path = file_name
      case Path():
        raise ValueError("file_name must contain a file name, not a directory")
      case str() if file_name.strip() and Path(file_name).name:
        self._file_name: Path = Path(file_name)
      case str():
        raise ValueError("file_name must be non-empty when provided as a str, "
                         "and correspond to a file not to a directory")
      case _:
        raise TypeError("file_name must be provided as a non-empty string or "
                        "a Path")

    match repeat_last:
      case bool():
        self._repeat_last: bool = repeat_last
      case _:
        raise TypeError("repeat_last must be a boolean")

    match delimiter:
      case str() if delimiter.strip():
        self._delimiter: str = delimiter
      case str():
        raise ValueError("The delimiter must be a non-empty string")
      case _:
        raise TypeError("The delimiter must be a non-empty string")

    # Eagerly extract the data and check its consistency
    array = np.loadtxt(self._file_name, delimiter=self._delimiter, ndmin=2)
    if len(array.shape) != 2:
      raise ValueError(f"The file {self._file_name} should contain a 2D array "
                       f"with two columns")
    if array.shape[1] != 2:
      raise ValueError(f"The file {self._file_name} should contain exactly two"
                       f"columns !")
    if np.any(np.isnan(array)):
      raise ValueError(f"The file {self._file_name} contains NaN values")
    if not np.all(np.isfinite(array)):
      raise ValueError(f"The file {self._file_name} contains infinite values")
    self._timestamps: np.ndarray = array[:, 0]
    self._values: np.ndarray = array[:, 1]
    if not np.all(self._timestamps[:-1] < self._timestamps[1:]):
      raise ValueError("The timestamp values are not sorted in "
                       "chronological order")

    self._warned: bool = False
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float] | None:
    """Interpolates the file's value at ``dt``.

    Before the first timestamp, the first value is returned. Once the file is
    exhausted, a warning is logged once and the configured ``repeat_last``
    behavior applies.
    """

    # Warn when the file is exhausted
    if dt > self._timestamps[-1] and not self._warned:
      if self._repeat_last:
        self.log(logging.WARNING, f"Exhausted the command from the file "
                                  f"{self._file_name}, returning the last "
                                  f"provided value from now on")
      else:
        self.log(logging.WARNING, f"Exhausted the command from the file "
                                  f"{self._file_name}, no longer returning "
                                  f"values")
      self._warned = True

    if dt > self._timestamps[-1] and not self._repeat_last:
      return None
    
    return {self._label: float(np.interp(dt, self._timestamps, self._values))}

  def reset(self) -> None:
    """Clears the exhausted-file warning for the next State entry."""

    self.log(logging.DEBUG, "FromFile Output was reset")
    self._warned = False
