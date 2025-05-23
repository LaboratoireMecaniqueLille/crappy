# coding: utf-8

import numpy as np
from typing import Optional
import logging
from tkinter import TclError

from .meta_block import Block
from .._global import OptionalModule

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)


class Grapher(Block):
  """This Block can display data in a 2D graph in a persistent way.
  
  The graph is displayed in an independent window. It iis updated each time new
  data is received from upstream Blocks. The displayed data can come from 
  different Blocks.
  
  The user can choose which labels are plotted on the `x` and `y` axes. It is
  therefore possible to plot a label versus time, or a label versus another 
  label. A single graph is displayed, but multiple curves can be plotted on
  this graph.
  
  The Grapher Block is known for being very CPU-intensive. For displaying only 
  the last values of given labels, the :class:`~crappy.blocks.LinkReader` and 
  :class:`~crappy.blocks.Dashboard` Blocks are simpler solutions that go much
  easier on the CPU. 
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               *labels: tuple[str, str],
               length: int = 0,
               freq: Optional[float] = 2,
               max_pt: int = 20000,
               window_size: tuple[int, int] = (8, 8),
               window_pos: Optional[tuple[int, int]] = None,
               interp: bool = True,
               backend: str = "TkAgg",
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      *labels: Each :obj:`tuple` corresponds to a curve to plot, and should
        contain two values: the first will be the label of the `x` values, the
        second the label of the `y` values. There's no limit to the number of
        curves. Note that all the curves are displayed in a same graph.
      length: If `0` the graph is static and displays all data from the start
        of the assay. Else, only displays the last ``length`` received chunks,
        and drops the previous ones.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      max_pt: The maximum number of points displayed on the graph. When
        reaching this limit, the Block deletes one point out of two to avoid
        using too much memory and CPU.

        .. versionchanged:: 2.0.0 renamed from *maxpt* to *max_pt*
      window_size: The size of the graph, in inches.
      window_pos: The position of the graph in pixels. The first value is for
        the `x` direction, the second for the `y` direction. The origin is the
        top-left corner. Works with multiple screens.
      interp: If :obj:`True`, the data points are linked together by straight
        lines. Else, only the points are displayed.
      backend: The :mod:`matplotlib` backend to use. Performance may vary
        according to the chosen backend. Also, every backend may not be
        available depending on your machine.
      display_freq: if :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.6
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0

    Example:
      ::

        graph = Grapher(('t(s)', 'F(N)'), ('t(s)', 'def(%)'))

      will plot a dynamic graph with two lines plot (:math:`F=f(t)` and
      :math:`def=f(t)`).
      ::

        graph = Grapher(('def(%)', 'F(N)'), length=0)

      will plot a static graph.
      ::

        graph = Grapher(('t(s)', 'F(N)'), length=30)

      will plot a dynamic graph displaying the last 30 chunks of data.
    """

    super().__init__()
    self.niceness = 10
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._length = length
    self._max_pt = max_pt
    self._window_size = window_size
    self._window_pos = window_pos
    self._interp = interp
    self._backend = backend

    self._labels = labels

    self._ax = None
    self._canvas = None
    self._figure = None
    self._lines = None
    self._factor = None
    self._counter = None
    self._clear_button = None

  def prepare(self) -> None:
    """Configures the figure for displaying data."""

    # Switch to the required backend
    if self._backend:
      self.log(logging.INFO, f"Setting matplotlib backend to {self._backend}")
      plt.switch_backend(self._backend)

    # Create the figure and the subplot
    self._figure = plt.figure(figsize=self._window_size)
    self._canvas = self._figure.canvas
    self._ax = self._figure.add_subplot(111)
    self._figure.canvas.mpl_connect('key_press_event', self._on_press)
    self._ax.set_title('(Press c to clear the graph)',
                       fontsize='small', loc='right')

    # Add the lines or the dots
    self._lines = []
    for _ in self._labels:
      if self._interp:
        self._lines.append(self._ax.plot([], [])[0])
      else:
        self._lines.append(self._ax.plot([], [], 'o', markersize=3)[0])

    # Keep only 1/factor points on each line
    self._factor = [1 for _ in self._labels]
    # Count to drop exactly 1/factor points, no more and no less
    self._counter = [0 for _ in self._labels]

    # Add the legend
    legend = [y for _, y in self._labels]
    plt.legend(legend)
    plt.xlabel(', '.join(set(x for x, _ in self._labels)))
    plt.ylabel(', '.join(set(y for _, y in self._labels)))

    # Add a grid
    plt.grid()

    # Set the dimensions if required
    if self._window_pos:
      mng = plt.get_current_fig_manager()
      mng.window.wm_geometry("+%s+%s" % self._window_pos)

    # Ready to show the window
    self.log(logging.INFO, "Configured the matplotlib window, displaying it")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.001)

  def loop(self) -> None:
    """Receives the upcoming data, puts in the display buffer and updates the
    graph."""

    # Receives the data sent by the upstream blocks
    if self.freq >= 10:
      # Assuming that above 10Hz the data won't saturate the links
      data = self.recv_all_data_raw()
    else:
      # Below 10Hz, making sure to flush the pipes at least every 0.1s
      data = self.recv_all_data_raw(delay=1 / 2 / self.freq,
                                    poll_delay=min(0.1, 1 / 2 / self.freq))

    update = False  # Should the graph be updated ?

    # For each curve, looking for the corresponding labels in the received data
    for i, (lx, ly) in enumerate(self._labels):
      x, y = None, None
      for dic in data:
        if lx in dic and ly in dic:
          # Found the corresponding data, getting the new data according to the
          # current resampling factors
          dx = dic[lx][self._factor[i] - self._counter[i] -
                       1::self._factor[i]]
          dy = dic[ly][self._factor[i] - self._counter[i] -
                       1::self._factor[i]]
          # Recalculating the counter
          self._counter[i] = (self._counter[i] +
                              len(dic[lx])) % self._factor[i]
          # Adding the new points to the arrays
          x = np.hstack((self._lines[i].get_xdata(), dx))
          y = np.hstack((self._lines[i].get_ydata(), dy))
          # the graph will need to be updated
          update = True
          # As we found the data, no need to search any further
          break

      # In case no matching labels were found, aborting for this curve
      if x is None:
        continue

      # Adjusting the number of points to remain below the length limit
      if self._length and len(x) >= self._length:
        x = x[-self._length:]
        y = y[-self._length:]

      # Dividing the number of points by two to remain below the maxpt limit
      elif len(x) > self._max_pt:
        self.log(logging.INFO, f"Too many points on the graph "
                               f"{i} ({len(x)}>{self._max_pt})")
        x, y = x[::2], y[::2]
        self._factor[i] *= 2
        self.log(logging.INFO, f"Resampling factor is now {self._factor[i]}")

      # Finally, updating the data on the graph
      self.log(logging.DEBUG, f"Graph data for labels {lx}, {ly}: {x}, {y}")
      self._lines[i].set_xdata(x)
      self._lines[i].set_ydata(y)

    # Updating the graph if necessary
    if update:
      self.log(logging.DEBUG, "Updating the graph")
      self._ax.relim()
      self._ax.autoscale()
      try:
        self._canvas.draw()
      except TclError:
        pass
      self._canvas.flush_events()

  def finish(self) -> None:
    """Closes all the opened :mod:`matplotlib` windows."""

    self.log(logging.INFO, "Closing all matplotlib windows")
    plt.close("all")

  def _on_press(self, event) -> None:
    """Callback catching the keyboard press events.

    When called, resets the display by emptying the data buffers.
    """

    if event.key == 'c':
      for line in self._lines:
        line.set_xdata([])
        line.set_ydata([])
      self.factor = [1 for _ in self._labels]
      self.counter = [0 for _ in self._labels]

      self.log(logging.INFO, "Cleared the matplotlib window")
