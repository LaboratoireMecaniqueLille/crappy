# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Reader Reader
# @{

## @file reader.py
# @brief Read and print the input Link.
#
# @author Victor Couty
# @version 0.1
# @date 13/07/2016
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
      if d:
        s = ""
        if self.name:
          s += self.name+" "
        s += "got: "+str(d)
        print(s)
