# coding: utf-8

from time import time
import numpy as np
import pathlib
import logging

from .meta_path import Path
from ... import _validation as check


class Custom(Path):
  """Generates a custom Path from a text file, until the file is exhausted.

  The file can be in any text format, including the most common `.csv` and
  `.txt` extensions.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               file_name: str | pathlib.Path,
               delimiter: str = ',') -> None:
    """Loads the file and sets the arguments.

    The stop condition is simply to reach the last timestamp given in the
    file.

    Args:
      file_name: Path to the file containing the information on the Generator
        Path. Can be either a :obj:`str` or a :obj:`pathlib.Path`. The file
        must contain two columns: the first one containing timestamps (starting
        from 0), the other one containing the values.

        .. versionchanged:: 2.0.0 renamed from *filename* to *file_name*
      delimiter: The delimiter between columns in the file, usually a coma.
    
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    match file_name:
      case pathlib.Path() if file_name.name.strip():
        pass
      case pathlib.Path():
        raise ValueError("file_name must contain a file name, not a directory")
      case str() if file_name.strip() and pathlib.Path(file_name).name.strip():
        file_name = pathlib.Path(file_name)
      case str():
        raise ValueError("file_name must be non-empty when provided as a str, "
                         "and correspond to a file not to a directory")
      case _:
        raise TypeError("file_name must be provided as a non-empty string or "
                        "a Path")
    if file_name.is_dir():
      raise IsADirectoryError(f"file_name points to a directory: {file_name}")

    check.string(delimiter, 'delimiter', empty=True)
    if len(delimiter) != 1 or delimiter in ('\n', '\r'):
      raise ValueError("delimiter must be a single character other than "
                       "a newline")

    self.log(logging.DEBUG, f"Extracting data from file {file_name}")
    array = np.loadtxt(file_name, delimiter=delimiter, ndmin=2)

    if array.ndim != 2 or array.shape[0] < 2:
      raise ValueError(f"The file {file_name} should contain a 2D array with "
                       f"two columns")

    if array.shape[1] != 2:
      raise ValueError(f'The file {file_name} should contain exactly two'
                       f'columns !')

    if not np.all(np.isfinite(array)):
      raise ValueError("The file must contain only finite values")

    self._timestamps = array[:, 0]
    self._values = array[:, 1]

    if not np.all(self._timestamps[:-1] <= self._timestamps[1:]):
      raise ValueError("The timestamp values are not sorted in "
                       "chronological order")

  def get_cmd(self, _: dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met.

    The value is interpolated from the given file.
    """

    t = time()
    if t - self.t0 > self._timestamps[-1]:
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration
    return float(np.interp(t - self.t0, self._timestamps, self._values))
