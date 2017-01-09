#coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup CompacterBlock CompacterBlock
# @{

## @file _compacterblock.py
# @brief Class to represent blocks that will group data by packets before sending it
#
# @authors Victor Couty
# @version 1.1
# @date 09/01/2017
from __future__ import print_function

from ._masterblock import MasterBlock
from collections import OrderedDict

import time

class CompacterBlock(MasterBlock):
  """
  A class to represent any block that will need to stream data by chunks
  Usually used for data acquisition at any frequency (unless it is under 1~10Hz)
  """
  def __init__(self,labels,compacter):
    print("[CompacterBlock] Got labels", labels,"comp=",compacter)
    MasterBlock.__init__(self)
    self.labels = labels
    self.compacter = compacter
    self.clear_results()

  def clear_results(self):
    """To clear the buffer"""
    self.results = OrderedDict.fromkeys(self.labels)
    for k in self.results.keys():
      self.results[k] = []

  def send(self,data):
    """Will actually send only once every self.compacter loops"""
    #t = time.time()
    #elapsed = t-self.t0
    #self.results[self.labels[0]].append(elapsed)
    for index,key in enumerate(self.labels):
      self.results[key].append(data[index])
    if len(self.results[self.labels[0]]) == self.compacter:
      MasterBlock.send(self,self.results)
      #print("sent",self.results)
      self.clear_results()
    
