#coding: utf-8

import numpy as np

from .modifier import Modifier


class Demux(Modifier):
  """
  Modifier to change a stream table into a dict with values (to plot streams).

  This modifier turns the array return by a streaming device into a
  dict with individual values (but only one per table). This allows
  attaching graphers to HF acquisition devices.

  Note:
    The table will be lost in the process.

  Args:
    - labels: The names of the labels to use for each column of the array.

  Kwargs:
    - stream (default: 'stream'): The name of the label containing the stream.
    - mean: If true, the returned value will be the average of the column
      else it will be the first value only.
    - time_label: The name of the label of the time table.

  """
  def __init__(self,*labels,**kwargs):
    Modifier.__init__(self)
    if len(labels) == 1 and isinstance(labels[0],list):
      self.labels = labels[0]
    else:
      self.labels = labels
    self.stream = kwargs.pop("stream","stream")
    self.mean = kwargs.pop("mean",False)
    self.time = kwargs.pop("time_label","t(s)")
    self.transpose = kwargs.pop("transpose",False)
    assert not kwargs,"Demux modifier got invalid kwarg:"+str(kwargs)
    if self.mean:
      self.evaluate = self.evaluate_mean
    else:
      self.evaluate = self.evaluate_nomean

  def evaluate(self):
    pass

  def evaluate_nomean(self,data):
    if 0 in data[self.stream].shape:
      return data
    for i,n in enumerate(self.labels):
      if self.transpose:
        data[n] = data[self.stream][i,0]
      else:
        data[n] = data[self.stream][0,i]
    del data[self.stream]
    try:
      data[self.time] = data[self.time][0]
    except Exception:
      pass
    return data

  def evaluate_mean(self,data):
    if 0 in data[self.stream].shape:
      return data
    for i,n in enumerate(self.labels):
      if self.transpose:
        data[n] = np.mean(data[self.stream][i,:])
      else:
        data[n] = np.mean(data[self.stream][:,i])
    del data[self.stream]
    data[self.time] = np.mean(data[self.time])
    return data
