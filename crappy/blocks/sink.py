# coding: utf-8

from .masterblock import MasterBlock

class Sink(MasterBlock):
  """
  Test block used to get data and do nothing
  """

  def __init__(self, *args, **kwargs):
    MasterBlock.__init__(self)

  def loop(self):
    self.drop()
