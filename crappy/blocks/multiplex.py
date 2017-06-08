#coding: utf-8

import numpy as np
from itertools import chain
#chain(*l) is MUCH faster than sum(l,[]) to concatenate lists

from .masterblock import MasterBlock

def interp(xp,yp,x):
  try:
    return np.interp(x,xp,yp)
  except TypeError:
    # Make a nearest interpolation for non numerical values
    a = range(len(yp))
    return yp[int(np.interp(x,xp,a)+.5)]

class Multiplex(MasterBlock):
  """
  This bloc is meant to interpolate data in order to return a data flux
  with a common time base
  It is really useful to save data from multiple sensors, it will
  then be easier to process at once.
  It will send data at a fixed frequency, but needs a delay to make
  sure all the required data for interpolation has already been processed
  """
  def __init__(self,key='t(s)',freq=200,delay=1):
    MasterBlock.__init__(self)
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
    self.delay = delay
    self.t = 0
    self.dt = 1/self.freq

  def begin(self):
    """
    We need to receive the first bit of data from each input to know the
    labels and make lists of what we will read in the main loop
    """
    for i,link in enumerate(self.inputs):
      r = link.recv_chunk()
      if not self.k in r: # This input don't have the timebase label!
        self.t_hist.append([])
        self.label_list.append([])
        continue
      self.t_hist.append(r[self.k]) # Append the time to this link's history
      labels = [] # To recap all the labels we will use at each loop
      for k in r:
        if k == self.k: # Ignore the timebase label (treated separatly)
          continue
        if k in chain(*self.label_list):
          print("WARNING, label %s comes from multiple inputs, ignoring one"%k)
          continue
        labels.append(k)
        self.hist[k] = r[k] # Put the data in the hist dict
      self.label_list.append(labels) # Add the label list of this link

  def loop(self):
    for i,l in enumerate(self.label_list):
      r = self.inputs[i].recv_chunk() # Get the data
      if not l:
        continue
      self.t_hist[i].extend(r[self.k]) # Add the time to this link's history
      for k in l:
        self.hist[k].extend(r[k]) # Add each data to their history
    while all([i and i[-1] > self.t for i in self.t_hist]): # Send all we can
      r = {self.k:self.t} # First data to return: our new timebase
      for i,l in enumerate(self.label_list):
        if not l:
          continue
        for k in l:
          r[k] = interp(self.t_hist[i],self.hist[k],self.t) # Interpolate
        # Delete the data we don't need anymore
        # May be a bit slow ?
        while len(self.t_hist[i]) > 2 and self.t_hist[i][1] < self.t:
          del self.t_hist[i][0]
          for k in l:
            del self.hist[k][0]
      self.send(r) # Send it!
      self.t += self.dt # And increment our timebase
