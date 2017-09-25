#coding: utf-8

import numpy as np

from .condition import Condition

class Demux(Condition):
  """
  Condition to change a stream table into a dict with values (to plot streams).

  This condition turns the array return by a streaming device into a
  dict with individual values (but only one per table). This allows
  attaching graphers to HF acquisition devices.
  Note that the table will be lost in the process.
  Args:
    labels: The names of the labels to use for each column of the array.
  kwargs:
    stream: The name of the label containing the stream (default:'stream')
    mean: If true, the returned value will be the average of the column
      else it will be the first value only
    time_label: The name of the label of the time table.
  """
  def __init__(self,*labels,**kwargs):
    Condition.__init__(self)
    if len(labels) == 1 and isinstance(labels[0],list):
      self.labels = labels[0]
    else:
      self.labels = labels
    self.stream =  kwargs.pop("stream","stream")
    self.mean = kwargs.pop("mean",False)
    self.time = kwargs.pop("time_label","t(s)")
    assert not kwargs,"Demux condition got invalid kwarg:"+str(kwargs)
    if self.mean:
      self.evaluate = self.evaluate_mean
    else:
      self.evaluate = self.evaluate_nomean

  def evaluate(self):
    pass

  def evaluate_nomean(self,data):
    for i,n in enumerate(self.labels):
      data[n] = data[self.stream][0,i]
    del data[self.stream]
    data[self.time] = data[self.time][0]
    return data

  def evaluate_mean(self,data):
    for i,n in enumerate(self.labels):
      data[n] = np.mean(data[self.stream][:,i])
    del data[self.stream]
    data[self.time] = np.mean(data[self.time])
    return data
