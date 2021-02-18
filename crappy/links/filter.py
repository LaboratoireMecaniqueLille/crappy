# coding: utf-8
from .condition import Condition
import numpy as np


class Filter(Condition):
  """
  Filters the signal.

  Note:
    This condition filters the selected data with a mean or a median.

  Parameters:
    - labels (list of str): List of all the labels of the data you want to filter.
    - mode {'median', 'mean'}: You can either filter with a mean or a median.
    - size (int, default: 10): Define on how many point you want to apply your
      filter.

  Returns:
    dict (OrderedDict): The dict contains the same values as the input,
    plus the filtered value, labeled as (name_of_the_input_label)_filtered.
  """
  def __init__(self, labels=[], mode="median", size=10):

    self.mode = mode
    self.size = size
    self.labels = labels
    self.FIFO = [[] for label in self.labels]

  # print self.FIFO

  def evaluate(self, value):
    for i, label in enumerate(self.labels):
      # print self.FIFO[i]
      self.FIFO[i].insert(0, value[label])
      if len(self.FIFO[i]) > self.size:
        self.FIFO[i].pop()
      if self.mode == "median":
        result = np.median(self.FIFO[i])
      elif self.mode == "mean":
        result = np.mean(self.FIFO[i])
      value[label + "_filtered"] = result
    return value
