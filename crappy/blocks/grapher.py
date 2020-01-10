# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from .masterblock import MasterBlock


def split(x,y):
  """
  Takes [x0,x1,x2,...,xn] and [y0,y1,y2,...,yn]

  returns [x0,x1,x1,x2,x2,...,xn,xn] and [y0,y0,y1,y1,y2,...,y(n-1),yn]
  This is to move only one axis at a time when plotting the list (see interp)
  """
  rx = sum([[i,i] for i in x],[])[1:]
  ry = sum([[i,i] for i in y],[])[:-1]
  return rx,ry


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

      optional: window_pos (tuple: (x_position, y_position), in PIXELS).
          Defines where on the screen the window grapher pops. Works with
          multiple screens. The origin is the top left corner. For instance,
          to have a graph that pops on the top right corner:
          window_pos=(1920, 0)

      optional: interp (bool, default=True)
          If True, the points of data will be linked by straight lines
          else, each value wil be displayed as constant until the next update
          In simple words,
          . ' will be linked like so _| if False and like so / if True

  Examples
  --------
      graph=Grapher(('t(s)','F(N)'),('t(s)','def(%)'))
          plot a dynamic graph with two lines plot(F=f(t) and def=f(t)).
      graph=Grapher(('def(%)','F(N)'),length=0)
          plot a static graph.
      graph=Grapher(('t(s)','F(N)'),length=30)
          plot a dynamic graph that will display the last 30 chunks of data
  """

  def __init__(self, *args, **kwargs):
    MasterBlock.__init__(self)
    self.niceness = 10
    self.length = kwargs.pop("length", 0)
    self.freq = kwargs.pop("freq", 2)
    self.maxpt = kwargs.pop("maxpt", 20000)
    self.window_size = kwargs.pop("window_size", (8, 8))
    self.window_pos = kwargs.pop("window_pos", None)
    self.interp = kwargs.pop("interp",True)
    self.backend = kwargs.pop("backend",None)
    if kwargs:
      raise AttributeError("Invalid kwarg(s) in Grapher: " + str(kwargs))
    self.labels = args

  def prepare(self):
    if self.backend:
      plt.switch_backend(self.backend)
    self.f = plt.figure(figsize=self.window_size)
    self.ax = self.f.add_subplot(111)
    self.lines = []
    for _ in self.labels:
      if self.interp:
        self.lines.append(self.ax.plot([], [])[0])
      else:
        self.lines.append(self.ax.step([], [])[0])
    # Keep only 1/factor points on each line
    self.factor = [1 for i in self.labels]
    # Count to drop exactly 1/factor points, no more and no less
    self.counter = [0 for i in self.labels]
    legend = [y for x, y in self.labels]
    plt.legend(legend, bbox_to_anchor=(-0.03, 1.02, 1.06, .102), loc=3,
               ncol=len(legend), mode="expand", borderaxespad=1)
    plt.xlabel(self.labels[0][0])
    plt.ylabel(self.labels[0][1])
    plt.grid()
    self.axclear = plt.axes([.8,.02,.15,.05])
    self.bclear = Button(self.axclear,'Clear')
    self.bclear.on_clicked(self.clear)

    if self.window_pos:
      mng = plt.get_current_fig_manager()
      mng.window.wm_geometry("+%s+%s" % self.window_pos)
    plt.draw()
    plt.pause(.001)

  def clear(self,event=None):
    for l in self.lines:
      l.set_xdata([])
      l.set_ydata([])
    self.factor = [1 for i in self.labels]
    self.counter = [0 for i in self.labels]

  def loop(self):
    # We need to recv data from all the links, but keep
    # ALL of the data, even with the same label (so not get_all_last)
    data = [l.recv_chunk() if l.poll() else {} for l in self.inputs]
    for i, (lx, ly) in enumerate(self.labels):
      x = 0 # So that if we don't find it, we do nothing
      for d in data:
        if lx in d and ly in d: # Find the first input with both labels
          dx = d[lx][self.factor[i]-self.counter[i]-1::self.factor[i]]
          dy = d[ly][self.factor[i]-self.counter[i]-1::self.factor[i]]
          self.counter[i] = (self.counter[i]+len(d[lx]))%self.factor[i]
          x = np.hstack((self.lines[i].get_xdata(), dx))
          y = np.hstack((self.lines[i].get_ydata(), dy))
          break
      if isinstance(x,int):
        break
      if self.length and len(x) >= self.length:
        # Remove the begining if the graph is dynamic
        x = x[-self.length:]
        y = y[-self.length:]
      elif len(x) > self.maxpt:
        # Reduce the number of points if we have to many to display
        print("[Grapher] Too many points on the graph {} ({}>{})".format(
          i,len(x),self.maxpt))
        x,y = x[::2], y[::2]
        self.factor[i] *= 2
        print("[Grapher] Resampling factor is now {}".format(self.factor[i]))
      self.lines[i].set_xdata(x)
      self.lines[i].set_ydata(y)
    self.ax.relim() # Update the window
    self.ax.autoscale_view(True, True, True)
    self.f.canvas.draw() # Update the graph
    self.f.canvas.flush_events()

  def finish(self):
    plt.close("all")
