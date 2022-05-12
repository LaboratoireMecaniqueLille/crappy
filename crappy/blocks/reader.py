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

    Block.__init__(self)
    self.freq = freq
    self.verbose = verbose

    index = self._instance_index()
    self._name = name if name is not None else f'Reader {index}'

  def loop(self) -> None:
    """Simply flushes the link and prints its data."""

    for link in self.inputs:
      data = link.recv_chunk(blocking=False)
      if data is not None:
        for dic in (dict(i) for i in zip(*([(key, value) for value in values]
                                           for key, values in data.items()))):
          print(f'{self._name} got: {dic}')

  @classmethod
  def _instance_index(cls) -> int:
    """Returns the index of the current instance."""

    cls._index += 1
    return cls._index
