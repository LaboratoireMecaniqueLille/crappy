# coding: utf-8

from __future__ import annotations
from datetime import timedelta
from time import time
from typing import Any, Optional
from collections.abc import Iterable
import logging

from .meta_block import Block
from .._global import OptionalModule

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)
mpl = OptionalModule('matplotlib', lazy_import=True)


class Text:
  """Displays a simple text line on the drawing.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               _: Canvas,
               coord: tuple[int, int],
               text: str,
               label: str,
               **__: str) -> None:
    """Sets the arguments.

    Args:
      _: The parent drawing Block.
      coord: The coordinates of the text on the drawing.
      text: The text to display.
      label: The label carrying the information for updating the text.
      **__: Other unused arguments.

    .. versionchanged:: 1.5.10
       now explicitly listing the *_*, *coord*, *text* and *label* arguments
    """

    x, y = coord
    self._text = text
    self._label = label

    self._txt = plt.text(x, y, text)

  def update(self, data: dict[str, float]) -> None:
    """Updates the text according to the received values."""

    if self._label in data:
      self._txt.set_text(self._text % data[self._label])


class DotText:
  """Like :class:`Text`, but with a colored dot to visualize a numerical value.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Dot_text* to *DotText*
  """

  def __init__(self,
               drawing: Canvas,
               coord: tuple[int, int],
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
      normalized on the ``crange`` of the Block and the dot will change
      color from blue to red depending on this value.
      
    .. versionchanged:: 1.5.10
       now explicitly listing the *drawing*, *coord*, *text* and *label*
       arguments
    """

    x, y = coord
    self._text = text
    self._label = label

    self._txt = plt.text(x + 40, y + 20, text, size=16)
    self._dot = plt.Circle(coord, 20)

    drawing.ax.add_artist(self._dot)
    low, high = drawing.color_range

    self._amp = high - low
    self._low = low

  def update(self, data: dict[str, float]) -> None:
    """Updates the text and the color dot according to the received values."""

    if self._label in data:
      self._txt.set_text(self._text % data[self._label])
      self._dot.set_color(mpl.cm.coolwarm((data[self._label] -
                                           self._low) / self._amp))


class Time:
  """Displays a time counter on the drawing, starting at the beginning of the
  test.

  .. versionadded:: 1.4.0
  """

  def __init__(self, drawing: Canvas, coord: tuple[int, int], **__) -> None:
    """Sets the arguments.

    Args:
      drawing: The parent drawing Block.
      coord: The coordinates of the time counter on the drawing.
      **__: Other unused arguments.

    .. versionchanged:: 1.5.10
       now explicitly listing the *drawing* and *coord* arguments
    """

    self._block = drawing
    x, y = coord

    self._txt = plt.text(x, y, "00:00", size=38)

  def update(self, _: dict[str, float]) -> None:
    """Updates the time counter, independently of the received values."""

    self._txt.set_text(str(timedelta(seconds=int(time() - self._block.t0))))


class Canvas(Block):
  """This Block allows displaying a real-time visual representation of data.

  It displays the data on top of a background image and updates it according to
  the values received through the incoming :class:`~crappy.links.Link`. The
  background image and the data overlay are displayed in a new window.

  It is possible to display a simple text, a time counter, or text associated
  with a color dot evolving depending on a predefined color bar and the
  received values.

  This Block is mostly useful for displaying a user-friendly and fine-tuned
  representation of data. For simpler displays, the
  :class:`~crappy.blocks.Dashboard`, :class:`~crappy.blocks.Grapher` and
  :class:`~crappy.blocks.LinkReader` Blocks should be preferred.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Drawing* to *Canvas*
  """

  def __init__(self,
               image_path: str,
               draw: Optional[Iterable[dict[str, Any]]] = None,
               color_range: tuple[float, float] = (20, 300),
               title: str = "Canvas",
               window_size: tuple[int, int] = (7, 5),
               backend: str = "TkAgg",
               freq: Optional[float] = 2,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      image_path: Path to the image that will be the background of the canvas,
        as a :obj:`str`.
      draw: An iterable (like a :obj:`list` or a :obj:`tuple`) of :obj:`dict`
        defining what to draw. See below for more details.
      color_range: A :obj:`tuple` containing the lowest and highest values for
        the color bar.

        .. versionchanged:: 1.5.10 renamed from *crange* to *color_range*
      title: The title of the window containing the drawing.
      window_size: The `x` and `y` dimension of the window, following
        :mod:`matplotlib` nomenclature.
      backend: The :mod:`matplotlib` backend to use.
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
          coordinates where the element should be displayed on the drawing.

        - ``text``: Mandatory for `'text'` and `'dot_text'` only, the text to
          display on the drawing. It must follow the %-formatting, and contain
          exactly one %-field. Ex: `'T0 = %f'`. This field will be updated
          using the value carried by ``label``.

        - ``label``: Mandatory for `'text'` and `'dot_text'` only, the label of
          the data to display. It will try to retrieve this data in the
          incoming Links. The ``text`` will then be updated with this data.
    """

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._image = image_path
    self._draw = [] if draw is None else list(draw)
    self.color_range = color_range
    self._title = title
    self._window_size = window_size
    self._backend = backend

    self._fig = None
    self.ax = None
    self._drawing_elements = None

  def prepare(self) -> None:
    """Initializes the different elements of the drawing."""

    self.log(logging.INFO, "Opening the drawing windows")

    # Initializing the window and the background image
    plt.switch_backend(self._backend)
    self._fig, self.ax = plt.subplots(figsize=self._window_size)
    image = self.ax.imshow(plt.imread(self._image), cmap=mpl.cm.coolwarm)
    image.set_clim(-0.5, 1)

    # Initializing the color bar
    cbar = self._fig.colorbar(image, ticks=[-0.5, 1], fraction=0.061,
                              orientation='horizontal', pad=0.04)
    cbar.set_label('Dot text values')
    cbar.ax.set_xticklabels(self.color_range)

    # Setting the title and the axes
    self.ax.set_title(self._title)
    self.ax.set_axis_off()

    # Adding the elements to the drawing
    self._drawing_elements = []
    for dic in self._draw:
      if dic['type'] == 'text':
        self._drawing_elements.append(Text(self, **dic))
      elif dic['type'] == 'dot_text':
        self._drawing_elements.append(DotText(self, **dic))
      elif dic['type'] == 'time':
        self._drawing_elements.append(Time(self, **dic))

  def loop(self) -> None:
    """Receives the latest data from upstream Blocks and updates the drawing
    accordingly."""

    if not (data := self.recv_last_data(fill_missing=False)):
      return

    for elt in self._drawing_elements:
      elt.update(data)
    self.log(logging.DEBUG, "Updating the drawing window")
    self._fig.canvas.draw()
    plt.pause(0.001)

  def finish(self) -> None:
    """Closes the window containing the drawing."""

    self.log(logging.INFO, "Closing the drawing windows")
    plt.close()
