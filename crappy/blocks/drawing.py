# coding: utf-8

from datetime import timedelta
from time import time
from typing import Tuple, List, Dict, Any, Optional

from .block import Block
from .._global import OptionalModule

try:
  import matplotlib.pyplot as plt
  from matplotlib import cm
except (ModuleNotFoundError, ImportError):
  plt = OptionalModule("matplotlib")
  cm = OptionalModule("matplotlib")


class Text:
  """Displays a simple text line on the drawing."""

  def __init__(self,
               _,
               coord: Tuple[int, int],
               text: str,
               label: str,
               **__: str) -> None:
    """Simply sets the args.

    Args:
      _: The parent drawing block.
      coord: The coordinates of the text on the drawing.
      text: The text to display.
      label: The label carrying the information for updating the text.
      **__: Other unused arguments.
    """

    x, y = coord
    self._text = text
    self._label = label

    self._txt = plt.text(x, y, text)

  def update(self, data: Dict[str, float]) -> None:
    """Updates the text according to the received values."""

    self._txt.set_text(self._text % data[self._label])


class Dot_text:
  """Like :class:`Text`, but with a colored dot to visualize a numerical value.
  """

  def __init__(self,
               drawing,
               coord: Tuple[int, int],
               text: str,
               label: str,
               **__: str) -> None:
    """Simply sets the args.

    Args:
      drawing: The parent drawing block.
      coord: The coordinates of the text and the color dot on the drawing.
      text: The text to display.
      label: The label carrying the information for updating the text and the
        color of the dot.
      **__: Other unused arguments.

        Important:
          The value received in label must be a numeric value. It will be
          normalized on the ``crange`` of the block and the dot will change
          color from blue to red depending on this value.
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

  def update(self, data: Dict[str, float]) -> None:
    """Updates the text and the color dot according to the received values."""

    self._txt.set_text(self._text % data[self._label])
    self._dot.set_color(cm.coolwarm((data[self._label] -
                                     self._low) / self._amp))


class Time:
  """Displays a time counter on the drawing, starting at the beginning of the
  test."""

  def __init__(self, drawing, coord: Tuple[int, int], **__) -> None:
    """Simply sets the args.

    Args:
      drawing: The parent drawing block.
      coord: The coordinates of the time counter on the drawing.
      **__: Other unused arguments.
    """

    self._block = drawing
    x, y = coord

    self._txt = plt.text(x, y, "00:00", size=38)

  def update(self, _: Dict[str, float]) -> None:
    """Updates the time counter, independently of the received values."""

    self._txt.set_text(str(timedelta(seconds=int(time() - self._block.t0))))


class Drawing(Block):
  """This block allows displaying a real-time visual representation of data.

  It displays the data on top of a background image and updates it according to
  the values received through the incoming links.

  It is possible to display simple text, a time counter, ot text associated
  with a color dot evolving depending on a predefined color bar and the
  received values.
  """

  def __init__(self,
               image: str,
               draw: Optional[List[Dict[str, Any]]] = None,
               color_range: Tuple[float, float] = (20, 300),
               title: str = "Drawing",
               window_size: Tuple[int, int] = (7, 5),
               backend: str = "TkAgg",
               freq: float = 2,
               verbose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      image: Path to the image that will be the background of the canvas, as a
        :obj:`str`.
      draw: A :obj:`list` of :obj:`dict` defining what to draw. See below for
        more details.
      color_range: A :obj:`tuple` containing the lowest and highest values for
        the color bar.
      title: The title of the window containing the drawing.
      window_size: The x and y dimension of the window, following
        :mod:`matplotlib` nomenclature.
      backend: The :mod:`matplotlib` backend to use.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, prints the looping frequency of the block.

    Note:
      - Information about the ``draw`` keys:

        - ``type``: Mandatory, the type of drawing to display. It can be either
          `'text'`, `'dot_text'` or `'time'`.

        - ``coord``: Mandatory, a :obj:`tuple` containing the `x` and `y`
          coordinates where the element should be displayed on the drawing.

        - ``text``: Mandatory for :class:`Text` and :class:`Dot_text` only, the
          text to display on the drawing. It must follow the %-formatting, and
          contain exactly one %-field. This field will be updated using the
          value carried by ``label``.

        - ``label``: Mandatory for :class:`Text` and :class:`Dot_text` only,
          the label of the data to display. It will try to retrieve this data
          in the incoming links. The ``text`` will then be updated with this
          data.
    """

    super().__init__()
    self.freq = freq
    self.verbose = verbose

    self._image = image
    self._draw = [] if draw is None else draw
    self.color_range = color_range
    self._title = title
    self._window_size = window_size
    self._backend = backend

  def prepare(self) -> None:
    """Initializes the different elements of the drawing."""

    # Initializing the window and the background image
    plt.switch_backend(self._backend)
    self._fig, self.ax = plt.subplots(figsize=self._window_size)
    image = self.ax.imshow(plt.imread(self._image), cmap=cm.coolwarm)
    image.set_clim(-0.5, 1)

    # Initializing the color bar
    cbar = self._fig.colorbar(image, ticks=[-0.5, 1], fraction=0.061,
                              orientation='horizontal', pad=0.04)
    cbar.set_label('Temperatures(C)')
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
        self._drawing_elements.append(Dot_text(self, **dic))
      elif dic['type'] == 'time':
        self._drawing_elements.append(Time(self, **dic))

  def loop(self) -> None:
    """Receives the latest data from upstream blocks and updates the drawing
    accordingly."""

    data = self.get_last()
    for elt in self._drawing_elements:
      elt.update(data)
    self._fig.canvas.draw()
    plt.pause(0.001)

  def finish(self) -> None:
    """Simply closes the window containing the drawing."""

    plt.close()
