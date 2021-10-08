# coding: utf-8

from .block import Block


class Reader(Block):
  """Reads and prints the input :ref:`Link`.

  Creates a reader that prints the input data continuously.
  """

  def __init__(self, reader_name: str = None) -> None:
    """Sets the arg and initializes the parent class.

    Args:
      reader_name (:obj:`str`, optional): If set, will be printed to identify
        the reader.
    """

    Block.__init__(self)
    self.reader_name = reader_name

  def loop(self) -> None:
    for i in self.inputs:
      d = i.recv_last()
      if d is not None:
        s = ""
        if self.reader_name:
          s += self.reader_name + " "
        s += "got: " + str(d)
        print(s)
