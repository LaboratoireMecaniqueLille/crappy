# coding: utf-8

from .block import Block


class Sink(Block):
  """Test block used to get data and do nothing."""

  def __init__(self, *_, **__):
    Block.__init__(self)

  def loop(self) -> None:
    self.drop()
