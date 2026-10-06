# coding: utf-8

from __future__ import annotations
from datetime import timedelta
from math import isfinite, isnan
from numbers import Real
from time import time
from typing import Any, TYPE_CHECKING, Literal
from collections.abc import Sequence
import logging
import locale
import os
import tkinter as tk
from pathlib import Path
from warnings import warn

from .meta_block import Block
from .._global import OptionalModule

mpl = OptionalModule('matplotlib', lazy_import=True)
mpl_figure = OptionalModule('matplotlib.figure', lazy_import=True)
mpl_image = OptionalModule('matplotlib.image', lazy_import=True)
mpl_patches = OptionalModule('matplotlib.patches', lazy_import=True)
backend_tkagg = OptionalModule('matplotlib.backends.backend_tkagg',
                               lazy_import=True)
QtCore = OptionalModule('PyQt6.QtCore', lazy_import=True)
QtGui = OptionalModule('PyQt6.QtGui', lazy_import=True)
QtWidgets = OptionalModule('PyQt6.QtWidgets', lazy_import=True)

if TYPE_CHECKING:
  from matplotlib import figure as mpl_figure
  from matplotlib import patches as mpl_patches
  from matplotlib import text as mpl_text
  from matplotlib.axes import Axes
  from matplotlib.backends import backend_tkagg
  from PyQt6 import QtWidgets, QtCore, QtGui

# Sampling points for a native Qt approximation of the coolwarm color map
_COOLWARM_COLORS = ((59, 76, 192), (98, 130, 234), (141, 176, 254),
                    (185, 208, 249), (221, 220, 220), (245, 196, 172),
                    (244, 152, 122), (221, 95, 75), (180, 4, 38))


class Text:
  """Displays a simple text line on the drawing.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               drawing: Canvas,
               coord: tuple[float, float],
               text: str,
               label: str,
               **__: str) -> None:
    """Sets the arguments.

    Args:
      drawing: The parent drawing Block.
      coord: The coordinates of the text on the drawing.
      text: The text to display.
      label: The label carrying the information for updating the text.
      **__: Other unused arguments.

    .. versionchanged:: 1.5.10
       now explicitly listing the *_*, *coord*, *text* and *label* arguments
    .. versionchanged:: 2.1.0 renamed the *_* argument to *drawing*
    """

    x, y = coord
    self._drawing: Canvas = drawing
    self._text = text
    self._label = label
    self._txt: mpl_text.Text | None = None
    self._qt_txt: QtWidgets.QGraphicsSimpleTextItem | None = None

    if drawing.backend == 'tkinter':
      assert drawing.ax is not None
      self._txt = drawing.ax.text(x, y, text)
    else:
      assert drawing.qt_scene is not None
      self._qt_txt = drawing.qt_scene.addSimpleText(text)
      assert self._qt_txt is not None
      self._qt_txt.setPos(x, y)
      self._qt_txt.setZValue(2)
      # Match Matplotlib's fixed screen font size while the image is scaled
      self._qt_txt.setFlag(
          QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
      self._set_font_size(10)

  def update(self, data: dict[str, float]) -> None:
    """Updates the text according to the received values."""

    if self._label in data:
      self._set_text(self._text % data[self._label])

  def _set_text(self, text: str) -> None:
    """Updates the text artist or native Qt item for this overlay."""

    if self._qt_txt is not None:
      self._qt_txt.setText(text)
    else:
      assert self._txt is not None
      self._txt.set_text(text)

  def _set_font_size(self, size: float) -> None:
    """Sets the font size in points and anchors Qt text at its baseline."""

    if self._qt_txt is not None:
      font = self._qt_txt.font()
      font.setPointSizeF(size)
      self._qt_txt.setFont(font)
      ascent = QtGui.QFontMetricsF(font).ascent()
      self._qt_txt.setTransform(QtGui.QTransform.fromTranslate(0, -ascent))
    else:
      assert self._txt is not None
      self._txt.set_fontsize(size)


class DotText(Text):
  """Like :class:`Text`, but with a colored dot to visualize a numerical value.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Dot_text* to *DotText*
  """

  def __init__(self,
               drawing: Canvas,
               coord: tuple[float, float],
               text: str,
               label: str,
               **__: str) -> None:
    """Sets the arguments.

    Args:
      drawing: The parent drawing Block.
      coord: The coordinates of the text and the color dot on the drawing.
      text: The text to display.
      label: The label carrying the information for updating the text and the
        color of the dot.
      **__: Other unused arguments.

    Important:
      The value received in label must be a numeric value. It will be
      normalized on the ``color_range`` of the Block and the dot will change
      color from blue to red depending on this value.
      
    .. versionchanged:: 1.5.10
       now explicitly listing the *drawing*, *coord*, *text* and *label*
       arguments
    """

    x, y = coord
    super().__init__(drawing, (x + 40, y + 20), text, label)
    self._set_font_size(16)
    self._dot: mpl_patches.Circle | None = None
    self._qt_dot: QtWidgets.QGraphicsEllipseItem | None = None

    if drawing.backend == 'tkinter':
      assert drawing.ax is not None
      self._dot = mpl_patches.Circle(coord, 20)
      assert self._dot is not None
      drawing.ax.add_artist(self._dot)
    else:
      assert drawing.qt_scene is not None
      self._qt_dot = drawing.qt_scene.addEllipse(
          x - 20, y - 20, 40, 40,
          QtGui.QPen(QtCore.Qt.PenStyle.NoPen),
          QtGui.QBrush(QtGui.QColor('#1f77b4')))
      assert self._qt_dot is not None
      self._qt_dot.setZValue(1)
    low, high = drawing.color_range

    self._amp = high - low
    self._low = low

  def update(self, data: dict[str, float]) -> None:
    """Updates the text and the color dot according to the received values."""

    if self._label in data:
      super().update(data)
      value = (data[self._label] - self._low) / self._amp
      if self._qt_dot is not None:
        self._qt_dot.setBrush(QtGui.QBrush(self._drawing.qt_color(value)))
      else:
        assert self._dot is not None
        self._dot.set_color(mpl.cm.coolwarm(value))


class Time(Text):
  """Displays a time counter on the drawing, starting at the beginning of the
  test.

  .. versionadded:: 1.4.0
  """

  def __init__(self,
               drawing: Canvas,
               coord: tuple[float, float],
               **__) -> None:
    """Sets the arguments.

    Args:
      drawing: The parent drawing Block.
      coord: The coordinates of the time counter on the drawing.
      **__: Other unused arguments.

    .. versionchanged:: 1.5.10
       now explicitly listing the *drawing* and *coord* arguments
    """

    self._block = drawing
    super().__init__(drawing, coord, '00:00', '')
    self._set_font_size(38)

  def update(self, data: dict[str, float]) -> None:
    """Updates the time counter, independently of the received values."""

    self._set_text(str(timedelta(seconds=int(time() - self._block.t0))))


class Canvas(Block):
  """This Block allows displaying a real-time visual representation of data.

  It displays the data on top of a background image and updates it according to
  the values received through the incoming :class:`~crappy.links.link.Link`.
  The background image and the data overlay are displayed in a new window.

  It is possible to display a simple text, a time counter, or text associated
  with a color dot evolving depending on a predefined color bar and the
  received values.

  The graphical interface uses PyQt6 by default, or :mod:`tkinter` when
  requested. The PyQt6 backend draws the image, overlays, and color bar using
  native Qt graphics. The Tkinter backend uses :mod:`matplotlib`. Both display
  the title above the image and the color bar below it. Closing the window does
  not stop the test.

  This Block is mostly useful for displaying a user-friendly and fine-tuned
  representation of data. For simpler displays, the
  :class:`~crappy.blocks.Dashboard`, :class:`~crappy.blocks.Grapher` and
  :class:`~crappy.blocks.LinkReader` Blocks should be preferred.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Drawing* to *Canvas*
  .. versionchanged:: 2.1.0 the default backend is now PyQt, no longer Tkinter
  """

  def __init__(self,
               image_path: str | Path,
               draw: Sequence[dict[str, Any]] | None = None,
               color_range: tuple[float, float] = (20, 300),
               title: str = "Canvas",
               window_size: tuple[float, float] = (7, 5),
               backend: Literal['tkinter', 'pyqt'] = 'pyqt',
               freq: float | None = 2,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      image_path: Path to the image that will be the background of the canvas,
        as a non-empty :obj:`str` or a :obj:`~pathlib.Path` object.
      draw: A sequence (like a :obj:`list` or a :obj:`tuple`) of :obj:`dict`
        defining what to draw. See below for more details.
      color_range: A :obj:`tuple` containing two distinct finite values for
        the color bar. The bounds are sorted from lowest to highest.

        .. versionchanged:: 1.5.10 renamed from *crange* to *color_range*
      title: The non-empty title of the window containing the drawing.
      window_size: The positive width and height of the drawing, in inches.
      backend: The library used for the graphical interface, either
        ``'pyqt'`` (the default) or ``'tkinter'``. The ``'pyqt'`` backend
        requires PyQt6 and provides native image rendering, text and dot
        overlays, and a blue-to-red color bar. The ``'tkinter'`` backend
        requires :mod:`matplotlib`.

        .. versionchanged:: 2.1.0 accepts ``'pyqt'`` and ``'tkinter'``
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0

    Note:
      - Information about the ``draw`` keys:

        - ``type``: Mandatory, the type of drawing to display. It can be either
          `'text'`, `'dot_text'` or `'time'`.

        - ``coord``: Mandatory, a :obj:`tuple` containing the `x` and `y`
          coordinates where the element should be displayed on the drawing,
          in image pixels. Both values must be finite numbers.

        - ``text``: Mandatory for `'text'` and `'dot_text'` only, the text to
          display on the drawing. It must follow the %-formatting, and contain
          exactly one %-field. Ex: `'T0 = %f'`. This field will be updated
          using the value carried by ``label``.

        - ``label``: Mandatory for `'text'` and `'dot_text'` only, the label of
          the data to display. It will try to retrieve this data in the
          incoming Links. The ``text`` will then be updated with this data.
    """

    warn("\nThe meaning of the backend argument for the Canvas Block was "
         "changed from Matplotlib backend selection to selection between"
         "'tkinter' and 'pyqt', without prior notice.\nThis change was "
         "implemented nevertheless because it is part of a larger migration "
         "to PyQt\nSet backend='tkinter' to switch back to the "
         "previous behavior.\nInstall PyQt6 to use the new Canvas "
         "interface.\n", UserWarning, stacklevel=2)

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match image_path:
      case Path() if image_path.name:
        self._image: Path = image_path
      case Path():
        raise ValueError("image_path must contain a file name, not a "
                         "directory")
      case str() if image_path.strip() and Path(image_path).name:
        self._image: Path = Path(image_path)
      case str():
        raise ValueError("image_path must be non-empty when provided as a "
                         "str, and correspond to a file not to a directory")
      case _:
        raise TypeError("image_path must be provided as a non-empty string or "
                        "a Path")

    match draw:
      case None:
        self._draw: list[dict[str, Any]] = list()
      case Sequence() if not isinstance(draw, (str, bytes)):
        self._draw: list[dict[str, Any]] = list()
        for element in draw:
          self._validate_element(element)
          self._draw.append(element.copy())
      case _:
        raise TypeError("draw must be a sequence of dictionaries or None")

    match color_range:
      case (Real() as low, Real() as high) if isinstance(color_range, tuple):
        if not isfinite(low) or not isfinite(high):
          raise ValueError("color_range must contain two finite numbers")
        if low == high:
          raise ValueError("The two values of color_range cannot be equal")
        self.color_range: tuple[float, float] = (min(low, high),
                                                 max(low, high))
      case _:
        raise TypeError("color_range must be a tuple of two finite numbers")

    match title:
      case str() if title.strip():
        self._title: str = title
      case str():
        raise ValueError("title must be a non-empty string")
      case _:
        raise TypeError("title must be a non-empty string")

    match window_size:
      case (Real() as width, Real() as height) if isinstance(window_size,
                                                             tuple):
        if (not isfinite(width) or not isfinite(height) or
            width <= 0 or height <= 0):
          raise ValueError("window_size must contain two finite positive "
                           "numbers")
        self._window_size: tuple[float, float] = (float(width), float(height))
      case _:
        raise TypeError("window_size must be a tuple of two positive numbers")

    match backend:
      case str() if backend in ('tkinter', 'pyqt'):
        self.backend: Literal['tkinter', 'pyqt'] = backend
      case str():
        raise ValueError("The backend must be either 'tkinter' or 'pyqt'")
      case _:
        raise TypeError("The backend must be either 'tkinter' or 'pyqt'")

    # Drawing elements use the selected backend's text and dot objects
    self._drawing_elements: list[Text | DotText | Time] = list()

    # Attributes related to tkinter
    self._fig: mpl_figure.Figure | None = None
    self.ax: Axes | None = None
    self._root: tk.Tk | None = None
    self._tk_canvas: backend_tkagg.FigureCanvasTkAgg | None = None

    # Attributes related to PyQt6
    self._qt_app: QtWidgets.QApplication | None = None
    self._qt_window: QtWidgets.QMainWindow | None = None
    self.qt_scene: QtWidgets.QGraphicsScene | None = None
    self._qt_view: QtWidgets.QGraphicsView | None = None
    self._qt_view_size: QtCore.QSize | None = None

  def prepare(self) -> None:
    """Initializes the different elements of the drawing."""

    if not self.inputs:
      raise IOError("The Canvas Block has no input Link!")
    if self.outputs:
      raise IOError("The Canvas Block does not accept output Links!")

    self.log(logging.INFO, "Opening the drawing window")

    if self.backend == 'tkinter':
      self._prepare_tkinter()
    else:
      self._prepare_pyqt()

  def loop(self) -> None:
    """Services the interface and updates the drawing with the latest data."""

    # Keep the window responsive even when no data is received
    if self.backend == 'pyqt':
      assert self._qt_app is not None
      assert self._qt_view is not None
      self._qt_app.processEvents()
      if self._qt_view.viewport().size() != self._qt_view_size:
        self._fit_qt_view()
    else:
      try:
        assert self._fig is not None
        self._fig.canvas.flush_events()
      except tk.TclError:
        return

    data = self.recv_last_data(fill_missing=False)
    if not data and not any(isinstance(elt, Time)
                            for elt in self._drawing_elements):
      return

    for elt in self._drawing_elements:
      elt.update(data)
    self.log(logging.DEBUG, "Updating the drawing window")
    if self.backend == 'pyqt':
      assert self._qt_app is not None
      self._qt_app.processEvents()
    else:
      try:
        assert self._fig is not None
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
      except tk.TclError:
        pass

  def finish(self) -> None:
    """Closes the drawing window, including after partial preparation.

    An existing Qt application is left available for its other windows.
    """

    self.log(logging.INFO, "Closing the drawing window")
    if getattr(self, '_qt_window', None) is not None:
      assert self._qt_window is not None
      self._qt_window.close()
    if getattr(self, '_qt_app', None) is not None:
      assert self._qt_app is not None
      self._qt_app.processEvents()
    try:
      if getattr(self, '_root', None) is not None:
        assert self._root is not None
        self._root.destroy()
    except tk.TclError:
      pass

  @staticmethod
  def _validate_element(element: dict[str, Any]) -> None:
    """Checks the type, coordinates, and required fields of an overlay."""

    match element:
      case dict() if 'type' in element and 'coord' in element:
        pass
      case dict():
        raise ValueError("All draw dictionaries must contain 'type' and "
                         "'coord' keys")
      case _:
        raise TypeError("All draw elements must be dictionaries")

    match element['type']:
      case str() if element['type'] in ('text', 'dot_text', 'time'):
        pass
      case str():
        raise ValueError("The 'type' key must be either 'text', 'dot_text', "
                         "or 'time'")
      case _:
        raise TypeError("The 'type' key must be a string")

    match element['coord']:
      case (Real() as x, Real() as y) if isinstance(element['coord'], tuple):
        if not isfinite(x) or not isfinite(y):
          raise ValueError("The coordinates must contain two finite numbers")
      case _:
        raise TypeError("The coordinates must be a tuple of two numbers")

    if element['type'] == 'time':
      return
    if 'text' not in element or 'label' not in element:
      raise ValueError("Text and dot_text elements must contain 'text' and "
                       "'label' keys")

    match element['text']:
      case str():
        pass
      case _:
        raise TypeError("The text must be a string")

    match element['label']:
      case str() if element['label'].strip():
        pass
      case str():
        raise ValueError("The label must be a non-empty string")
      case _:
        raise TypeError("The label must be a non-empty string")

  def _prepare_tkinter(self) -> None:
    """Creates the Tkinter drawing window."""

    self._root = tk.Tk()
    assert self._root is not None
    self._root.title(self._title)

    self._fig = mpl_figure.Figure(figsize=self._window_size)
    assert self._fig is not None
    self.ax = self._fig.add_subplot(111)
    self._tk_canvas = backend_tkagg.FigureCanvasTkAgg(
        self._fig, master=self._root)
    assert self._tk_canvas is not None
    self._tk_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True)
    self._set_tkinter_drawing()
    self._tk_canvas.draw()
    self._root.update()

  def _prepare_pyqt(self) -> None:
    """Creates the native Qt image scene, overlays, and color bar."""

    self._qt_app = self._get_application()
    self._qt_window = QtWidgets.QMainWindow()
    assert self._qt_window is not None
    self._qt_window.setWindowTitle(self._title)

    if not self._image.is_file():
      raise FileNotFoundError(f"Cannot find the background image: "
                              f"{self._image}")
    reader = QtGui.QImageReader(str(self._image))
    image = reader.read()
    if image.isNull():
      raise ValueError(f"Cannot read the background image {self._image}: "
                       f"{reader.errorString()}")

    self.qt_scene = QtWidgets.QGraphicsScene(self._qt_window)
    assert self.qt_scene is not None
    self.qt_scene.setSceneRect(QtCore.QRectF(image.rect()))
    background = self.qt_scene.addPixmap(QtGui.QPixmap.fromImage(image))
    assert background is not None
    background.setTransformationMode(
        QtCore.Qt.TransformationMode.SmoothTransformation)

    panel = QtWidgets.QWidget(self._qt_window)
    self._qt_window.setCentralWidget(panel)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(8)

    title = QtWidgets.QLabel(self._title)
    title.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    font = title.font()
    font.setPointSizeF(12)
    title.setFont(font)
    layout.addWidget(title)

    self._qt_view = QtWidgets.QGraphicsView(self.qt_scene, panel)
    assert self._qt_view is not None
    self._qt_view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    self._qt_view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    self._qt_view.setRenderHint(
        QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    self._qt_view.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self._qt_view.setVerticalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    layout.addWidget(self._qt_view, stretch=1)
    self._add_qt_color_bar(layout)

    self._add_drawing_elements()

    assert self._qt_app is not None
    screen = self._qt_app.primaryScreen()
    dpi_x = screen.logicalDotsPerInchX() if screen is not None else 96
    dpi_y = screen.logicalDotsPerInchY() if screen is not None else 96
    width = round(self._window_size[0] * dpi_x)
    height = round(self._window_size[1] * dpi_y)
    self._qt_window.resize(width, height)
    self._qt_window.show()
    self._qt_app.processEvents()
    self._fit_qt_view()
    self._qt_app.processEvents()

  def _set_tkinter_drawing(self) -> None:
    """Adds the background image, color bar, and overlays to the Tkinter
    drawing."""

    assert self.ax is not None
    assert self._fig is not None
    image = self.ax.imshow(mpl_image.imread(self._image), cmap=mpl.cm.coolwarm)
    image.set_clim(-0.5, 1)

    cbar = self._fig.colorbar(image, ticks=[-0.5, 1], fraction=0.061,
                              orientation='horizontal', pad=0.04)
    cbar.set_label('Dot text values')
    cbar.ax.set_xticklabels([str(value) for value in self.color_range])

    self.ax.set_title(self._title)
    self.ax.set_axis_off()
    self._add_drawing_elements()

  def _add_drawing_elements(self) -> None:
    """Creates the overlays using the selected backend's graphical items."""

    self._drawing_elements.clear()
    for element in self._draw:
      match element['type']:
        case 'text':
          self._drawing_elements.append(Text(self, **element))
        case 'dot_text':
          self._drawing_elements.append(DotText(self, **element))
        case 'time':
          self._drawing_elements.append(Time(self, **element))

  def _add_qt_color_bar(self, layout: QtWidgets.QVBoxLayout) -> None:
    """Adds a native gradient, range bounds, and caption below the image."""

    gradient = QtGui.QLinearGradient(0, 0, 256, 0)
    for index, color in enumerate(_COOLWARM_COLORS):
      gradient.setColorAt(index / (len(_COOLWARM_COLORS) - 1),
                          QtGui.QColor(*color))
    pixmap = QtGui.QPixmap(256, 16)
    painter = QtGui.QPainter(pixmap)
    try:
      painter.fillRect(pixmap.rect(), QtGui.QBrush(gradient))
    finally:
      painter.end()

    bar = QtWidgets.QLabel()
    bar.setPixmap(pixmap)
    bar.setScaledContents(True)
    bar.setFixedHeight(16)
    layout.addWidget(bar)

    bounds = QtWidgets.QHBoxLayout()
    bounds.addWidget(QtWidgets.QLabel(str(self.color_range[0])))
    bounds.addStretch()
    bounds.addWidget(QtWidgets.QLabel(str(self.color_range[1])))
    layout.addLayout(bounds)
    caption = QtWidgets.QLabel('Dot text values')
    caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(caption)

  @staticmethod
  def qt_color(value: float) -> QtGui.QColor:
    """Interpolates a blue-to-red color, clipping values outside the range.

    The native palette approximates Matplotlib's coolwarm color map. NaN
    values produce a transparent dot, as with the Tkinter backend.
    """

    if isnan(value):
      return QtGui.QColor(0, 0, 0, 0)
    position = min(1, max(0, value)) * (len(_COOLWARM_COLORS) - 1)
    index = min(int(position), len(_COOLWARM_COLORS) - 2)
    fraction = position - index
    low, high = _COOLWARM_COLORS[index:index + 2]
    return QtGui.QColor(*(round(start + fraction * (end - start))
                          for start, end in zip(low, high)))

  def _fit_qt_view(self) -> None:
    """Fits the image in the available viewport while preserving its aspect
    ratio."""

    assert self._qt_view is not None
    assert self.qt_scene is not None
    self._qt_view.fitInView(self.qt_scene.sceneRect(),
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    self._qt_view_size = self._qt_view.viewport().size()

  @staticmethod
  def _get_application() -> QtWidgets.QApplication:
    """Reuses a Qt widget application or creates one for this Block.

    OpenCV's bundled Qt plugin path is excluded during application creation,
    then restored. The numeric locale is also preserved for tkinter windows.
    """

    app = QtCore.QCoreApplication.instance()
    if app is not None:
      if not isinstance(app, QtWidgets.QApplication):
        raise RuntimeError("A non-widget Qt application already exists")
      return app

    original = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')
    if original is not None:
      paths = original.split(os.pathsep)
      filtered = [path for path in paths if
                  tuple(os.path.normpath(path).split(os.sep)[-3:]) !=
                  ('cv2', 'qt', 'plugins')]
      if filtered != paths:
        if filtered:
          os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(filtered)
        else:
          os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

    numeric_locale = locale.setlocale(locale.LC_NUMERIC)
    try:
      return QtWidgets.QApplication([])
    finally:
      locale.setlocale(locale.LC_NUMERIC, numeric_locale)
      if original is not None:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = original
