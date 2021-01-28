# coding: utf-8


from .block import Block


class Reader(Block):
  """
  Read and print the input Link.
  """
  def __init__(self, name=None):
    """
    Create a reader that prints the input data continuously

    Args:
        name: if set, will be printed to identify the reader
    """
    Block.__init__(self)
    self.name = name

  def loop(self):
    for i in self.inputs:
      d = i.recv_last()
      if d is not None:
        s = ""
        if self.name:
          s += self.name+" "
        s += "got: "+str(d)
        print(s)
