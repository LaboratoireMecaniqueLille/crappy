# coding: utf-8

import numpy as np
from typing import Union

from .modifier import Modifier


class Demux(Modifier):
  """Modifier to change a stream table into a :obj:`dict` with values (to plot
  streams).

  This modifier turns the array return by a streaming device into a :obj:`dict`
  with individual values (but only one per table). This allows attaching
  graphers to HF acquisition devices.

  Note:
    The table will be lost in the process.
  """

  def __init__(self,
               *labels: Union[str, list],
               stream: str = "stream",
               mean: bool = False,
               time_label: str = "t(s)",
               transpose: bool = False) -> None:
    """Sets the instance attributes.

    Args:
      *labels (:obj:`str`, :obj:`list`): The names of the labels to use for
        each column of the array. May be either a list of labels, or the labels
        given as separate arguments.
      stream (:obj:`str`, optional): The name of the label containing the
        stream.
      mean (:obj:`bool`, optional): If :obj:`True`, the returned value will be
        the average of the column.
      time_label (:obj:`str`, optional): The name of the label of the time
        table.
      transpose (:obj:`bool`, optional):
    """

    Modifier.__init__(self)
    if len(labels) == 1 and isinstance(labels[0], list):
      self.labels = labels[0]
    else:
      self.labels = labels
    self.stream = stream
    self.mean = mean
    self.time = time_label
    self.transpose = transpose
    if self.mean:
      self.evaluate = self.evaluate_mean
    else:
      self.evaluate = self.evaluate_nomean

  def evaluate(self) -> None:
    pass

  def evaluate_nomean(self, data: dict) -> dict:
    if 0 in data[self.stream].shape:
      return data
    for i, n in enumerate(self.labels):
      if self.transpose:
        data[n] = data[self.stream][i, 0]
      else:
        data[n] = data[self.stream][0, i]
    del data[self.stream]
    data[self.time] = data[self.time][0]
    return data

  def evaluate_mean(self, data: dict) -> dict:
    if 0 in data[self.stream].shape:
      return data
    for i, n in enumerate(self.labels):
      if self.transpose:
        data[n] = np.mean(data[self.stream][i, :])
      else:
        data[n] = np.mean(data[self.stream][:, i])
    del data[self.stream]
    data[self.time] = np.mean(data[self.time])
    return data
