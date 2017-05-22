# coding: utf-8
from __future__ import print_function

from .masterblock import MasterBlock

class Reader(MasterBlock):
  """
  Read and print the input Link.
  """
  def __init__(self, name=None):
    """
    Create a reader that prints k and the input data in continuous.

    Args:
        name: if set, will be printed to identify the reader
    """
    MasterBlock.__init__(self)
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
