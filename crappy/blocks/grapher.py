#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from .masterblock import MasterBlock
class Grapher(MasterBlock):
  """
  The grapher receive data from a block (via a Link) and plots it.

  Args:
      args : tuple
          tuples of the columns labels of input data for plotting.
          You can add as much as you want, depending on your performances.

      optional: length=x (int)
          number of chunks to data to be kept on the graph (default: 10)

          length=0 will create a static graph:
          add new values at every refresh. If there
          is too many data (> 20000), delete one out of 2
          to avoid memory overflow.

      optional: window_size (tuple: (width, height), in INCHES).
          Self explanatory ?
      freq: Defines the refresh rate of the grapher

  Examples
  --------
      graph=Grapher(('t(s)','F(N)'),('t(s)','def(%)'))
          plot a dynamic graph with two lines plot(F=f(t) and def=f(t)).
      graph=Grapher(('def(%)','F(N)'),length=0)
          plot a static graph.
      graph=Grapher(('t(s)','F(N)'),length=30)
          plot a dynamic graph that will display the last 30 chunks of data
  """
  def __init__(self,*args,**kwargs):
    MasterBlock.__init__(self)
    self.length = kwargs.pop("length",0)
    self.freq = kwargs.pop("freq",5)
    self.maxpt = kwargs.pop("maxpt",20000)
    if kwargs:
      raise AttributeError("Invalid kwarg(s) in Grapher: "+str(kwargs))
    self.labels = args

  def prepare(self):
    self.f = plt.figure(figsize=(8,8))
    self.ax = self.f.add_subplot(111)
    self.lines = []
    for _ in self.labels:
      self.lines.append(self.ax.plot([],[])[0])
    legend = [y for x,y in self.labels]
    plt.legend(legend, bbox_to_anchor=(-0.03, 1.02, 1.06, .102), loc=3,
                    ncol=len(legend), mode="expand", borderaxespad=1)
    plt.grid()

  def loop(self):
    data = self.get_all_last()
    for i,(lx,ly) in enumerate(self.labels):
      x = np.hstack((self.lines[i].get_xdata(),data[lx]))
      y = np.hstack((self.lines[i].get_ydata(),data[ly]))
      if self.length and len(x) >= self.length:
        x = x[-self.length:]
        y = y[-self.length:]
      elif len(x) > self.maxpt:
        x = x[::2]
        y = y[::2]
      self.lines[i].set_xdata(x)
      self.lines[i].set_ydata(y)
    self.ax.autoscale_view(True,True,True)
    self.f.canvas.draw()
    self.ax.relim()
    plt.pause(.01)
