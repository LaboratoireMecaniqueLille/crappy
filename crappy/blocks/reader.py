# coding: utf-8

from typing import Optional
import logging
from .block import Block


class Reader(Block):
  """Reads and displays the data flowing through the input :ref:`Link`."""

  _index = 0

  def __init__(self,
               name: Optional[str] = None,
               freq: float = 50,
               verbose: bool = False,
               debug: bool = False) -> None:
    """Sets the arg and initializes the parent class.

    Args:
      name: If set, will be displayed to identify the reader.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, the looping frequency will be displayed every
        2s.
    """

    super().__init__()
    self.freq = freq
    self.verbose = verbose
    self.log_level = logging.DEBUG if debug else logging.INFO

    self._reader_name = name if name is not None else f'Reader ' \
                                                      f'{self._get_index()}'

  def __new__(cls, *args, **kwargs):
    """"""

    cls._index += 1
    return super().__new__(cls)

  @classmethod
  def _get_index(cls) -> int:
    """Returns the current number of instantiates Links, as an :obj:`int`."""

    return cls._index

  def loop(self) -> None:
    """Simply flushes the link and displays its data."""

    data = self.recv_all_data_raw()
    for link_data in data:
      for dic in (dict(i) for i in
                  zip(*([(key, value) for value in values]
                        for key, values in link_data.items()))):
        self.log(logging.INFO, f'{self._reader_name} got: {dic}')
