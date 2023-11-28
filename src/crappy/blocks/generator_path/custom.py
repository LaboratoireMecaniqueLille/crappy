# coding: utf-8

from time import time
from numpy import loadtxt, interp
from typing import Dict, Union
import pathlib
import logging

from .meta_path import Path


class Custom(Path):
  """Generates a custom Path from a text file, until the file is exhausted.

  The file can be in any text format, including the most common `.csv` and
  `.txt` extensions.
  """

  def __init__(self,
               file_name: Union[str, pathlib.Path],
               delimiter: str = ',') -> None:
    """Loads the file and sets the arguments.

    The stop condition is simply to reach the last timestamp given in the
    file.

    Args:
      file_name: Path to the file containing the information on the Generator
        Path. Can be either a :obj:`str` or a :obj:`pathlib.Path`. The file
        must contain two columns: the first one containing timestamps (starting
        from 0), the other one containing the values.
      delimiter: The delimiter between columns in the file, usually a coma.
    """

    super().__init__()

    self.log(logging.DEBUG, f"Extracting data from file {file_name}")
    array = loadtxt(pathlib.Path(file_name), delimiter=delimiter)

    if array.shape[1] != 2:
      raise ValueError(f'The file {file_name} should contain exactly two'
                       f'columns !')

    self._timestamps = array[:, 0]
    self._values = array[:, 1]

  def get_cmd(self, _: Dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met.

    The value is interpolated from the given file.
    """

    t = time()
    if t - self.t0 > self._timestamps[-1]:
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration
    return float(interp(t - self.t0, self._timestamps, self._values))
