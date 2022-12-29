# coding: utf-8

from typing import Optional
from .block import Block


class Reader(Block):
  """Reads and prints the data flowing through the input :ref:`Link`."""

  _index = 0

  def __init__(self,
               name: Optional[str] = None,
               freq: float = 50,
               verbose: bool = False) -> None:
    """Sets the arg and initializes the parent class.

    Args:
      name: If set, will be printed to identify the reader.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, the looping frequency will be printed every 2s.
    """

    super().__init__()
    self.freq = freq
    self.verbose = verbose

    self._name = name if name is not None else f'Reader {self._get_index()}'

  def __new__(cls, *args, **kwargs):
    """"""

    cls._index += 1
    return super().__new__(cls)

  @classmethod
  def _get_index(cls) -> int:
    """Returns the current number of instantiates Links, as an :obj:`int`."""

    return cls._index

  def loop(self) -> None:
    """Simply flushes the link and prints its data."""

    data = self.recv_all_data_raw()
    for link_data in data:
      for dic in (dict(i) for i in
                  zip(*([(key, value) for value in values]
                        for key, values in link_data.items()))):
        print(f'{self._name} got: {dic}')
