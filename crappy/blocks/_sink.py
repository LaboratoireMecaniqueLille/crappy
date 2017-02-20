# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Sink Sink
# @{

## @file _sink.py
# @brief Receives data (test block)
# @author Victor Couty
# @version 0.1
# @date 05/10/2016

from _masterblock import MasterBlock


class Sink(MasterBlock):
  """
  Test block used to get data and do nothing
  """

  def __init__(self, *args, **kwargs):
    MasterBlock.__init__(self)

  def loop(self):
    self.drop()
