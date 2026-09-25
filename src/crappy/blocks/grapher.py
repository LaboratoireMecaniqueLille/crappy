# coding: utf-8

from __future__ import annotations
from collections import deque
from itertools import chain
import logging
from math import isfinite
from numbers import Real
from time import monotonic
from tkinter import TclError
from typing import TYPE_CHECKING, Literal
import numpy as np
import os
from warnings import warn

from .meta_block import Block
from .._global import OptionalModule

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)
pg = OptionalModule('pyqtgraph', lazy_import=True)

if TYPE_CHECKING:
  from matplotlib.figure import Figure
  from matplotlib.backend_bases import FigureCanvasBase
  from matplotlib.axes._axes import Axes
  from matplotlib.lines import Line2D
  from PyQt6.QtWidgets import QApplication
  from PyQt6.QtGui import QShortcut
  from pyqtgraph.widgets.PlotWidget import PlotWidget
  from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem

MPL_BACKENDS = ('QtAgg', 'ipympl', 'GTK3Agg', 'GTK4Agg', 'macosx', 'TkAgg',
                'nbAgg', 'WebAgg', 'GTK3Cairo', 'GTK4Cairo', 'wxAgg')
PYQT_BACKENDS = ('PyQt6', 'PySide6', 'PyQt5', 'PySide2')


class Grapher(Block):
  """This Block can display data in a 2D graph in a persistent way.

  The graph is displayed in an independent window and is refreshed at a
  configurable frequency. The displayed data can come from different Blocks.

  The user can choose which labels are plotted on the `x` and `y` axes. It is
  therefore possible to plot a label versus time, or a label versus another
  label. A single graph is displayed, but multiple curves can be plotted on
  this graph.

  The Matplotlib plotter can be very CPU-intensive. PyQtGraph is the default
  because it refreshes substantially faster while using less CPU. For
  displaying only the latest values of given labels, the
  :class:`~crappy.blocks.LinkReader` and :class:`~crappy.blocks.Dashboard`
  Blocks are simpler solutions.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.1.0 Added :external+pyqtgraph:doc:`PyQtGraph <index>`
     as a supported plotter, and made it the default.
  """

  def __init__(self,
               *labels: tuple[str, str],
               plotter: Literal['mpl', 'pyqt'] = 'pyqt',
               upd_freq: float | None = 2.0,
               length: int | None = None,
               max_pt: int | None = 20000,
               window_size: tuple[float, float] = (8.0, 8.0),
               window_pos: tuple[int, int] | None = None,
               interp: bool = True,
               backend: str | None = None,
               freq: float | None = 20.0,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      *labels: Each :obj:`tuple` corresponds to a curve to plot, and should
        contain two values: the first will be the label of the `x` values, the
        second the label of the `y` values. There's no limit to the number of
        curves. Note that all the curves are displayed in a same graph.
      plotter: The library in charge of plotting the graph, either `'mpl'` for
        :mod:`matplotlib` or `'pyqt'` for
        :external+pyqtgraph:doc:`PyQtGraph <index>`. Matplotlib is the
        historical plotter for this Block, but is CPU-intensive and generally
        limited to a 10-20 Hz refresh rate. PyQtGraph is the newer default, it
        uses less CPU, is interactive, and can reach refresh rates up to
        100 Hz.

        .. versionadded:: 2.1.0
      upd_freq: The target refresh frequency for the displayed graph, as a
        :obj:`float`. It cannot exceed the target looping frequency of the
        Block (``freq`` argument). If :obj:`None`, the graph is refreshed at
        every loop. This value is a target that might not be achieved in real
        conditions.
      length: If :obj:`None` the graph is static and displays all data from the
        start of the test, subject to the resampling configured by ``max_pt``.
        Else, only displays the last ``length`` received points, and drops the
        previous ones. When setting ``length``, ``max_pt`` must be set to
        :obj:`None` because the two limits are mutually exclusive.

        .. versionchanged:: 2.1.0 can now take the value :obj:`None`, which is
           the new default
      max_pt: The maximum number of points displayed on the graph. When
        exceeding this limit, the Block deletes one point out of two to avoid
        using too much memory and CPU. If :obj:`None` (not recommended), all
        the points are kept no matter how many there are. Cannot be set at the
        same time as ``length``.

        .. versionchanged:: 2.0.0 renamed from *maxpt* to *max_pt*
        .. versionchanged:: 2.1.0 can now take the value :obj:`None`
      window_size: The positive width and height of the graph, in inches.
      window_pos: The position of the graph in pixels, as two integers. The
        first value is for the `x` direction, the second for the `y` direction.
        The origin is the top-left corner. Negative coordinates can be used for
        screens located to the left of or above the primary screen.
      interp: If :obj:`True`, the data points are linked together by straight
        lines. Else, only the points are displayed.
      backend: The :mod:`matplotlib` backend, or the Qt binding used by
        :external+pyqtgraph:doc:`PyQtGraph <index>`. If :obj:`None`, lets the
        selected plotter choose automatically. Accepted Matplotlib values are
        `'QtAgg'`, `'ipympl'`, `'GTK3Agg'`, `'GTK4Agg'`, `'macosx'`, `'TkAgg'`,
        `'nbAgg'`, `'WebAgg'`, `'GTK3Cairo'`, `'GTK4Cairo'`, and `'wxAgg'`.
        Accepted PyQtGraph values are `'PyQt6'`, `'PySide6'`, `'PyQt5'`, and
        `'PySide2'`. The requested backend or binding must be installed on the
        machine.

        .. versionchanged:: 2.1.0 changed default from `'TkAgg'` to :obj:`None`
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible. It cannot be lower than the target refresh
        frequency (``upd_freq``) that drives the frequency at which the
        display is refreshed.

        .. versionchanged:: 2.1.0 changed the default frequency from 2 to 20
      display_freq: If :obj:`True`, displays the looping frequency of the
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

      will plot a dynamic graph with two curves (:math:`F=f(t)` and
      :math:`def=f(t)`).
      ::

        graph = Grapher(('def(%)', 'F(N)'), length=None)

      will plot a static graph.
      ::

        graph = Grapher(('t(s)', 'F(N)'), length=30, max_pt=None)

      will plot a dynamic graph displaying the last 30 data points.
    """

    warn("The new 'plotter' and 'upd_freq' arguments were added to the "
         "Grapher Block without prior notice, and the defaults for 'freq' and "
         "'backend' were also changed.\nSet plotter='mpl' and backend='TkAgg' "
         "to use the previous display settings.\nThese changes we implemented "
         "nevertheless because they significantly increase performance",
         UserWarning, stacklevel=2)

    super().__init__()
    self.niceness = 10
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Checking the validity of the provided arguments
    match plotter:
      case _ if plotter in ('mpl', 'pyqt'):
        self._plotter: Literal['mpl', 'pyqt'] = plotter
      case _:
        raise ValueError("The plotter must be either 'mpl' or 'pyqt'")

    match upd_freq:
      case None:
        self._upd_freq: float | None = None
      case Real() if not isfinite(upd_freq) or upd_freq <= 0:
        raise ValueError("upd_freq must be a finite strictly positive float "
                         "or None")
      case Real() if self.freq is not None and upd_freq > self.freq:
        raise ValueError("upd_freq cannot exceed the target freq")
      case Real():
        self._upd_freq: float | None = float(upd_freq)
      case _:
        raise TypeError("upd_freq must be a strictly positive float or None")

    match labels:
      case ():
        raise ValueError("No labels to plot were provided")
      case (*items,) if (all(isinstance(item, tuple) for item in items) and
                         all(len(item) == 2 for item in items) and
                         all((isinstance(label, str) and label)
                             for label in chain.from_iterable(labels))):
        self._graph_labels: tuple[tuple[str, str], ...] = labels
      case _:
        raise ValueError("The labels to plot must be given as tuples of "
                         "exactly two non-empty strings")

    match length:
      case None:
        self._length: int | None = None
      case int() if length > 0:
        self._length: int | None = length
      case int():
        raise ValueError("length must be a strictly positive integer or None")
      case _:
        raise TypeError("length must be a strictly positive integer or None")

    match max_pt:
      case None:
        self._max_pt: int | None = None
      case int() if max_pt > 0:
        self._max_pt: int | None = max_pt
      case int():
        raise ValueError("max_pt must be a strictly positive integer or None")
      case _:
        raise TypeError("max_pt must be a strictly positive integer or None")

    match self._length, self._max_pt:
      case (None, _) | (_, None):
        pass
      case _:
        raise ValueError("Only one of length and max_pt can be set at a time")

    match window_size:
      case (Real() as width,
            Real() as height) if (isinstance(window_size, tuple) and
                                  isfinite(width) and isfinite(height)
                                  and width > 0 and height > 0):
        self._window_size: tuple[float, float] = (float(width), float(height))
      case (Real(), Real()):
        raise ValueError("window_size must contain two finite positive "
                         "numbers")
      case _:
        raise TypeError("window_size must be a tuple of two positive numbers")

    match window_pos:
      case None:
        self._window_pos: tuple[int, int] | None = None
      case (int(), int()):
        self._window_pos: tuple[int, int] | None = window_pos
      case _:
        raise TypeError("window_pos must be a tuple of two integers or None")

    match interp:
      case bool():
        self._interp: bool = interp
      case _:
        raise TypeError("interp must be a boolean")

    match backend:
      case None:
        self._backend: str | None = backend
      case _ if plotter == 'mpl' and backend in MPL_BACKENDS:
        self._backend: str | None = backend
      case _ if plotter == 'pyqt' and backend in PYQT_BACKENDS:
        self._backend: str | None = backend
      case _ if plotter == 'mpl':
        raise ValueError(f"The possible backends with plotter='mpl' are "
                         f"{MPL_BACKENDS}")
      case _ if plotter == 'pyqt':
        raise ValueError(f"The possible backends with plotter='pyqt' are "
                         f"{PYQT_BACKENDS}")

    # Attributes related to Matplotlib
    self._ax: Axes | None = None
    self._canvas: FigureCanvasBase | None = None
    self._figure: Figure | None = None
    self._lines: list[Line2D] = list()

    # Attributes related to buffered graph data
    self._data: list[list[deque[float]]] = list()
    self._buf: list[list[list[float]]] = list()
    self._factor: list[int] = list()
    self._counter: list[int] = list()
    self._last_upd: float = float('-inf')
    self._refresh_count: int = 0
    self._last_refresh_rate: float = monotonic()

    # Attributes related to pyqtgraph
    self._qt_app: QApplication | None = None
    self._qt_plot: PlotWidget | None = None
    self._qt_curves: list[PlotDataItem] = list()
    self._qt_clear_shortcut: QShortcut | None = None

  def prepare(self) -> None:
    """Configures the figure for displaying data."""

    if not self.inputs:
      raise IOError("The Grapher Block has no input Link!")
    if self.outputs:
      raise IOError("The Grapher Block does not accept output Links!")

    # Add the data buffers
    self._data = [[deque(maxlen=self._length), deque(maxlen=self._length)]
                  for _ in self._graph_labels]
    self._buf = [[[], []] for _ in self._graph_labels]
    # Keep only 1/factor points on each line
    self._factor = [1 for _ in self._graph_labels]
    # Count raw points to preserve the resampling phase across refreshes
    self._counter = [0 for _ in self._graph_labels]

    # Prepare according to the selected plotter
    if self._plotter == 'pyqt':
      self._prepare_pyqtgraph()
    elif self._plotter == 'mpl':
      self._prepare_mpl()

  def loop(self) -> None:
    """Receives the upcoming data, puts in the display buffer and updates the
    graph."""

    if self._plotter == 'mpl' and (self._ax is None or self._canvas is None):
      raise RuntimeError("The Grapher must be prepared before loop is called")
    if self._plotter == 'pyqt' and (self._qt_app is None
                                    or self._qt_curves is None):
      raise RuntimeError("The Grapher must be prepared before loop is called")

    # Service the Qt GUI on every Block loop
    if self._plotter == 'pyqt':
      self._qt_app.processEvents()

    # Receives the data sent by the upstream Blocks
    data = self.recv_all_data_raw()

    # For each couple of labels, check for a Link containing both entries
    for (lx, ly), buf in zip(self._graph_labels, self._buf):
      for dic in data:
        # Store the received values until the next display refresh
        if lx in dic and ly in dic:
          if len(dic[lx]) != len(dic[ly]):
            raise RuntimeError(f"The received values for labels {lx} and "
                               f"{ly} have different lengths")
          buf[0].extend(dic[lx])
          buf[1].extend(dic[ly])
          # The curve is generated from the first Link carrying both labels
          break

    # Case when it's too early to refresh the graph
    if (self._upd_freq is not None and
        monotonic() - self._last_upd < 1 / self._upd_freq):
      # Display the refresh frequency if requested
      if self.display_freq:
        self._print_freq(refreshed=False)
      return

    # Should the graph be updated for each curve
    update = [False for _ in self._graph_labels]

    # Append the data to the overall buffer, and clear the looping buffers
    for i, ((lx, ly), buf, data, factor, counter) in enumerate(zip(
        self._graph_labels, self._buf, self._data, self._factor,
        self._counter)):
      if buf[0] and buf[1]:
        start = (-counter) % factor
        self._counter[i] = counter + len(buf[0])
        data[0].extend(buf[0][start::factor])
        data[1].extend(buf[1][start::factor])
        buf[0].clear()
        buf[1].clear()
        update[i] = True

        # Divide the number of points by two to remain below the max_pt limit
        while self._max_pt is not None and len(data[0]) > self._max_pt:
          self.log(logging.INFO, f"Too many points on the graph "
                                 f"{(lx, ly)} ({len(data[0])}>{self._max_pt})")
          data[0] = deque(list(data[0])[::2])
          data[1] = deque(list(data[1])[::2])
          self._factor[i] *= 2
          self.log(logging.INFO, f"Resampling factor is now {self._factor[i]}")

    # Update the Matplotlib buffers when necessary
    if self._plotter == 'mpl' and any(update):
      for upd, line, data, (lx, ly) in zip(update, self._lines, self._data,
                                           self._graph_labels):
        if upd:
          self.log(logging.DEBUG, f"Update graph data for labels {lx}, {ly}")
          line.set_data(data[0], data[1])

    # Update the PyQtGraph buffers and graph when necessary
    elif self._plotter == 'pyqt' and any(update):
      for upd, curve, data, (lx, ly) in zip(update, self._qt_curves,
                                            self._data, self._graph_labels):
        if upd:
          self.log(logging.DEBUG, f"Update graph data for labels {lx}, {ly}")
          curve.setData(np.asarray(data[0]), np.asarray(data[1]))
      # Service the Qt GUI to redraw data promptly
      self._qt_app.processEvents()

    # Update the Matplotlib graph when necessary
    if self._plotter == 'mpl' and any(update):
      self.log(logging.DEBUG, "Updating the graph")
      self._ax.relim()
      self._ax.autoscale()
      try:
        self._canvas.draw()
        self._canvas.flush_events()
      except TclError:
        pass

    # Update the last update time
    self._last_upd = monotonic()

    # Display the refresh frequency if requested
    if self.display_freq:
      self._print_freq(refreshed=True)

  def finish(self) -> None:
    """Closes the plotting window owned by this Block."""

    if self._qt_plot is not None:
      self.log(logging.INFO, "Closing the pyqtgraph window")
      self._qt_plot.close()
    if self._qt_app is not None:
      self._qt_app.processEvents()

    if self._figure is not None:
      self.log(logging.INFO, "Closing the matplotlib window")
      plt.close(self._figure)

  def _prepare_mpl(self) -> None:
    """The path ``prepare`` follows when the selected plotter is Matplotlib"""

    # Switch to the required backend
    if self._backend is not None:
      self.log(logging.INFO, f"Setting matplotlib backend to {self._backend}")
      plt.switch_backend(self._backend)

    # Create the figure and the subplot
    self._figure = plt.figure(figsize=self._window_size)
    self._canvas = self._figure.canvas
    self._ax = self._figure.add_subplot(111)
    self._canvas.mpl_connect('key_press_event', self._on_press_mpl)
    self._ax.set_title('(Press c to clear the graph)',
                       fontsize='small', loc='right')

    # Add the lines or the dots
    self._lines = [self._ax.plot([], [])[0] if self._interp
                   else self._ax.plot([], [], 'o', markersize=3)[0]
                   for _ in self._graph_labels]

    # Add the legend
    legend = [y for _, y in self._graph_labels]
    self._ax.legend(legend)
    self._ax.set_xlabel(', '.join(set(x for x, _ in self._graph_labels)))
    self._ax.set_ylabel(', '.join(set(y for _, y in self._graph_labels)))

    # Add a grid
    self._ax.grid()

    # Set the dimensions if required
    if self._window_pos is not None:
      manager = self._canvas.manager
      window = getattr(manager, 'window', None)
      x_pos, y_pos = self._window_pos
      if window is None:
        self.log(logging.WARNING, "The selected matplotlib backend does not "
                                  "support positioning its window")
      elif hasattr(window, 'wm_geometry'):
        window.wm_geometry(f"{x_pos:+d}{y_pos:+d}")
      elif hasattr(window, 'move'):
        window.move(x_pos, y_pos)
      elif hasattr(window, 'SetPosition'):
        window.SetPosition((x_pos, y_pos))
      else:
        self.log(logging.WARNING, "Cannot position the matplotlib window "
                                  "with the selected backend")

    # Ready to show the window
    self.log(logging.INFO, "Configured the matplotlib window, displaying it")
    self._figure.tight_layout()
    plt.show(block=False)
    plt.pause(.001)

  def _prepare_pyqtgraph(self) -> None:
    """The path ``prepare`` follows when the selected plotter is PyQtGraph"""

    if self._backend is not None:
      self.log(logging.INFO, f"Setting pyqtgraph backend to {self._backend}")
      os.environ["PYQTGRAPH_QT_LIB"] = self._backend

    # Configure the global PyQt app
    self._qt_app = pg.mkQApp("Crappy Grapher")

    # Manage color palette across light and dark themes
    palette = self._qt_app.palette()
    fg = palette.color(pg.Qt.QtGui.QPalette.ColorRole.Text)
    bg = palette.color(pg.Qt.QtGui.QPalette.ColorRole.Base)

    # Configure the plot widget
    self._qt_plot = pg.PlotWidget(background=bg)

    # Customize the plot widget with a grid, a legend, and axis titles
    self._qt_plot.addLegend(pen=pg.mkPen(fg), labelTextColor=fg)
    self._qt_plot.showGrid(x=True, y=True, alpha=0.5)
    self._qt_plot.setLabel('bottom', ', '.join(set(x for x, _
                                                   in self._graph_labels)))
    self._qt_plot.setLabel('left', ', '.join(set(y for _, y
                                                 in self._graph_labels)))
    self._qt_plot.setTitle("(Press c to clear the graph)", color=fg,
                           size="10pt")
    self._qt_plot.getPlotItem().titleLabel.setAttr('justify', 'right')

    # Make the axes and text stand out better
    for name in ('bottom', 'left'):
      axis = self._qt_plot.getAxis(name)
      axis.setStyle(maxTickLevel=0, tickAlpha=0.5)
      axis.setPen(pg.mkPen(fg))
      axis.setTickPen(pg.mkPen(fg))
      axis.setTextPen(pg.mkPen(fg))

    # Register a data clearing when pressing the 'c' key
    self._qt_clear_shortcut = pg.Qt.QtGui.QShortcut(
        pg.Qt.QtGui.QKeySequence("c"),
        self._qt_plot)
    self._qt_clear_shortcut.activated.connect(self._on_press_pyqt)

    # The curve that will actually plot the data
    self._qt_curves = [self._qt_plot.plot(
        [], [],
        name=ly,
        pen=None if not self._interp else pg.mkPen(pg.intColor(i), width=1),
        symbol='o' if not self._interp else None,
        symbolSize=3) for i, (_, ly) in enumerate(self._graph_labels)]

    # Resize the window to the requested dimension
    screen = self._qt_app.primaryScreen()
    self._qt_plot.resize(
        round(self._window_size[0] * screen.logicalDotsPerInchX()),
        round(self._window_size[1] * screen.logicalDotsPerInchY()))

    # Position the window as requested
    if self._window_pos is not None:
      self._qt_plot.move(*self._window_pos)

    # Ready to display the plotting window
    self.log(logging.INFO, "Configured the pyqtgraph window, displaying it")
    self._qt_plot.show()
    self._qt_app.processEvents()

  def _on_press_mpl(self, event) -> None:
    """Callback catching the keyboard press events.

    When called, resets the display by emptying the data buffers.
    """

    if event.key == 'c':
      for line in self._lines:
        line.set_data([], [])
      self._data = [[deque(maxlen=self._length), deque(maxlen=self._length)]
                    for _ in self._graph_labels]
      self._buf = [[[], []] for _ in self._graph_labels]
      self._factor = [1 for _ in self._graph_labels]
      self._counter = [0 for _ in self._graph_labels]

      # Request a new rendering of the Figure
      if self._canvas is not None:
        try:
          self._canvas.draw_idle()
        except TclError:
          pass

      self.log(logging.INFO, "Cleared the matplotlib window")

  def _on_press_pyqt(self) -> None:
    """Callback catching the press events on the 'c' key.

    When called, resets the display by emptying the data buffers.
    """

    for curve in self._qt_curves:
      curve.clear()
    self._data = [[deque(maxlen=self._length), deque(maxlen=self._length)]
                  for _ in self._graph_labels]
    self._buf = [[[], []] for _ in self._graph_labels]
    self._factor = [1 for _ in self._graph_labels]
    self._counter = [0 for _ in self._graph_labels]

    # Request a new rendering of the plot
    self._qt_app.processEvents()

    self.log(logging.INFO, "Cleared the pyqtgraph window")

  def _print_freq(self, refreshed: bool) -> None:
    """Periodically logs the achieved refresh frequency.

    The count can differ from the Block's loop frequency because loops that do
    not refresh the display are excluded.

    Args:
      refreshed: Whether an image was handled during the current loop.
    """

    self._refresh_count += int(refreshed)
    t = monotonic()
    if t - self._last_refresh_rate > 2:
      self.log(logging.INFO,
               f"Grapher refresh per second: "
               f"{self._refresh_count / (t - self._last_refresh_rate)}")
      self._last_refresh_rate = t
      self._refresh_count = 0
