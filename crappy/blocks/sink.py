# coding: utf-8

from .block import Block


class Sink(Block):
  """Test block used to get data and do nothing."""

  def __init__(self,
               verbose: bool = False,
               freq: float = 10,
               *_,
               **__) -> None:
    Block.__init__(self)
    self.verbose = verbose
    self.freq = freq

  def loop(self) -> None:
    self.drop()
