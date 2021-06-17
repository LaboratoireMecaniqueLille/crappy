# coding: utf-8

import numpy as np

from .block import Block
from .._global import OptionalModule

try:
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Button
except (ModuleNotFoundError, ImportError):
  plt = OptionalModule("matplotlib")
  Button = OptionalModule("matplotlib")


class Grapher(Block):
  """The grapher receive data from a block (via a :ref:`Link`) and plots it."""

  def __init__(self,
               *labels,
               length=0,
               freq=2,
               maxpt=20000,
               window_size=(8, 8),
               window_pos=None,
               interp=True,
               backend="TkAgg"):
    """Sets the args and initializes the parent class.

    Args:
      *labels (:obj:`tuple`): Tuples of the columns labels of input data for
        plotting. You can add as much as you want, depending on your
        performances. The first value is the `x` label, the second is the `y`
        label.
      length (:obj:`int`, optional): If `0` the graph is static and displays
        all data from the start of the assay. Else only displays the last
        ``length`` received chunks, and drops the previous ones.
      freq (:obj:`float`, optional): The refresh rate of the graph. May cause
        high memory consumption if set too high.
      maxpt (:obj:`int`, optional): The maximum number of points displayed on
        the graph. When reaching this limit, the block deletes one point out of
        two but this is almost invisible to the user.
      window_size (:obj:`tuple`, optional): The size of the graph, in inches.
      window_pos (:obj:`tuple`, optional): The position of the graph in pixels.
        The first value is for the `x` direction, the second for the `y`
        direction. The origin is the top left corner. Works with multiple
        screens.
      interp (:obj:`bool`, optional): If :obj:`True`, the points of data will
        be linked to the following by straight lines. Else, each value wil be
        displayed as constant until the next update.
      backend (:obj:`int`, optional): The :mod:`matplotlib` backend to use.

    Example:
      ::

        graph = Grapher(('t(s)', 'F(N)'), ('t(s)', 'def(%)'))

      will plot a dynamic graph with two lines plot (`F=f(t)` and `def=f(t)`).
      ::

        graph = Grapher(('def(%)', 'F(N)'), length=0)

      will plot a static graph.
      ::

        graph = Grapher(('t(s)', 'F(N)'), length=30)

      will plot a dynamic graph displaying the last 30 chunks of data.
    """

    Block.__init__(self)
    self.niceness = 10
    self.length = length
    self.freq = freq
    self.maxpt = maxpt
    self.window_size = window_size
    self.window_pos = window_pos
    self.interp = interp
    self.backend = backend

    self.labels = labels

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
    self.factor = [1 for _ in self.labels]
    # Count to drop exactly 1/factor points, no more and no less
    self.counter = [0 for _ in self.labels]
    legend = [y for x, y in self.labels]
    plt.legend(legend, bbox_to_anchor=(-0.03, 1.02, 1.06, .102), loc=3,
               ncol=len(legend), mode="expand", borderaxespad=1)
    plt.xlabel(self.labels[0][0])
    plt.ylabel(self.labels[0][1])
    plt.grid()
    self.axclear = plt.axes([.8, .02, .15, .05])
    self.bclear = Button(self.axclear, 'Clear')
    self.bclear.on_clicked(self.clear)

    if self.window_pos:
      mng = plt.get_current_fig_manager()
      mng.window.wm_geometry("+%s+%s" % self.window_pos)
    plt.draw()
    plt.pause(.001)

  def clear(self, event=None):
    for line in self.lines:
      line.set_xdata([])
      line.set_ydata([])
    self.factor = [1 for _ in self.labels]
    self.counter = [0 for _ in self.labels]

  def loop(self):
    # We need to recv data from all the links, but keep
    # ALL of the data, even with the same label (so not get_all_last)
    data = self.recv_all_delay()
    for i, (lx, ly) in enumerate(self.labels):
      x, y = 0, 0  # So that if we don't find it, we do nothing
      for d in data:
        if lx in d and ly in d:  # Find the first input with both labels
          dx = d[lx][self.factor[i]-self.counter[i]-1::self.factor[i]]
          dy = d[ly][self.factor[i]-self.counter[i]-1::self.factor[i]]
          self.counter[i] = (self.counter[i]+len(d[lx])) % self.factor[i]
          x = np.hstack((self.lines[i].get_xdata(), dx))
          y = np.hstack((self.lines[i].get_ydata(), dy))
          break
      if isinstance(x, int):
        break
      if self.length and len(x) >= self.length:
        # Remove the beginning if the graph is dynamic
        x = x[-self.length:]
        y = y[-self.length:]
      elif len(x) > self.maxpt:
        # Reduce the number of points if we have to many to display
        print("[Grapher] Too many points on the graph {} ({}>{})".format(
          i, len(x), self.maxpt))
        x, y = x[::2], y[::2]
        self.factor[i] *= 2
        print("[Grapher] Resampling factor is now {}".format(self.factor[i]))
      self.lines[i].set_xdata(x)
      self.lines[i].set_ydata(y)
    self.ax.relim()  # Update the window
    self.ax.autoscale_view(True, True, True)
    self.f.canvas.draw()  # Update the graph
    self.f.canvas.flush_events()

  def finish(self):
    plt.close("all")
