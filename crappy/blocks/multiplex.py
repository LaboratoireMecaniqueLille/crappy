# coding: utf-8

import numpy as np
from itertools import chain
# chain(*l) is MUCH faster than sum(l, []) to concatenate lists

from .block import Block


def interp(xp, yp, x):
  try:
    return np.interp(x, xp, yp)
  except (TypeError, ValueError):
    # Make a nearest interpolation for non numerical values
    a = range(len(yp))
    return yp[int(np.interp(x, xp, a) + .5)]


class Multiplex(Block):
  """This block interpolates data.

  It is used to read data and return the reading from multiple sensors at the
  same instants and a constant frequency. This block uses linear interpolation
  whenever possible, else the nearest neighbor.

  Warning:
    This block  needs a delay to make sure all the required data for
    interpolation has already been received, so do not use this block as the
    input of a decision block!
  """

  def __init__(self, key='t(s)', freq=200):
    """Sets the args and initializes the parent class.

    Args:
      key (:obj:`str`, optional): The label carrying the time information.
      freq (:obj:`float`, optional): The block will loop at this frequency.
    """

    Block.__init__(self)
    self.k = key
    self.freq = freq
    self.hist = dict()
    # Dict, keys = labels, values = list of the values
    self.t_hist = []
    # ith element is a list containing a list of the timestamps from
    # the ith link
    self.label_list = []
    # ith element is a list containing a list of the labels to get from
    # the ith link
    self.t = 0
    self.dt = 1 / self.freq

  def begin(self):
    """We need to receive the first bit of data from each input to know the
    labels and make lists of what we will read in the main loop."""

    for i, link in enumerate(self.inputs):
      r = link.recv_chunk()
      if self.k not in r:  # This input don't have the timebase label!
        self.t_hist.append([])
        self.label_list.append([])
        continue
      self.t_hist.append(r[self.k])  # Append the time to this link's history
      labels = []  # To recap all the labels we will use at each loop
      for k in r:
        if k == self.k:  # Ignore the timebase label (treated separately)
          continue
        if k in chain(*self.label_list):
          print("WARNING, label %s comes from multiple inputs, "
                "ignoring one" % k)
          continue
        labels.append(k)
        self.hist[k] = r[k]  # Put the data in the hist dict
      self.label_list.append(labels)  # Add the label list of this link

  def get_data(self):
    for i, l in enumerate(self.label_list):
      r = self.inputs[i].recv_chunk()  # Get the data
      if not l:
        continue
      self.t_hist[i].extend(r[self.k])  # Add the time to this link's history
      for k in l:
        self.hist[k].extend(r[k])  # Add each data to their history

  def send_data(self):
    while all([i and i[-1] > self.t for i in self.t_hist]) \
      and self.t_hist:  # Send all we can
      r = {self.k: self.t}  # First data to return: our new timebase
      for i, l in enumerate(self.label_list):
        if not l:
          continue
        for k in l:
          r[k] = interp(self.t_hist[i], self.hist[k], self.t)  # Interpolate
        # To know until when we can delete
        last = int(interp(self.t_hist[i], range(len(self.t_hist[i])),
          self.t + self.dt))
        # Delete the data we don't need anymore
        self.t_hist[i] = self.t_hist[i][last:]
        for k in l:
          self.hist[k] = self.hist[k][last:]
      self.send(r)  # Send it!
      self.t += self.dt  # And increment our timebase

  def loop(self):
    self.get_data()
    self.send_data()

  def finish(self):
    self.send_data()
    while any([len(k) > 1 for k in self.t_hist]):
      r = {self.k: self.t}  # First data to return: our new timebase
      for i, l in enumerate(self.label_list):
        if not l:
          continue
        for k in l:
          r[k] = interp(self.t_hist[i], self.hist[k], self.t)  # Interpolate
        # To know until when we can delete
        last = int(interp(self.t_hist[i], range(len(self.t_hist[i])),
          self.t + self.dt))
        # Delete the data we don't need anymore
        self.t_hist[i] = self.t_hist[i][last:]
        for k in l:
          self.hist[k] = self.hist[k][last:]
      self.send(r)  # Send it!
      self.t += self.dt  # And increment our timebase
