# coding: utf-8

from datetime import timedelta
from time import time

from .block import Block
from .._global import OptionalModule

try:
  import matplotlib.pyplot as plt
  from matplotlib import cm
except (ModuleNotFoundError, ImportError):
  plt = OptionalModule("matplotlib")
  cm = OptionalModule("matplotlib")


# ======= Visual objects =========
# These classes represent all that can be drawn on the canvas
# Common arg:
#  drawing: The drawing itself. It is used to access its attributes when needed
# Common kwargs:
#   coord: The coordinates of the object on the Canvas
#
# Update method will be called frequently by the Drawing block, you can
# define here what it will do on each update
# It gives the argument "data" containing all the latest data received
# by the block


class Text(object):
  """A simple text line."""

  def __init__(self, *_, **kwargs):
    """Sets the args.

    Args:
      *_: Contains the :class:`Drawing` object, unused.
      **kwargs: Contains the coordinates, the text and the label to be
        displayed. Also contains the type of drawing, but unused.
    """

    self.coord = kwargs['coord']
    self.text = kwargs['text']
    self.label = kwargs['label']

    self.txt = plt.text(self.coord[0], self.coord[1], self.text)

  def update(self, data):
    self.txt.set_text(self.text % data[self.label])


class Dot_text(object):
  """Like :class:`Text`, but with a colored dot to visualize a numerical value.
  """

  def __init__(self, drawing, **kwargs):
    """Sets the args.

    Args:
      drawing: The :class:`Drawing` object.
      **kwargs: Contains the coordinates, the text and the label to be
        displayed. Also contains the type of drawing, but unused.

        Important:
          The value received in label must be a numeric value. It will be
          normalized on the ``crange`` of the block and the dot will change
          color from blue to red depending on this value.
    """

    self.coord = kwargs['coord']
    self.text = kwargs['text']
    self.label = kwargs['label']

    self.txt = plt.text(self.coord[0] + 40, self.coord[1] + 20, self.text,
                        size=16)
    self.dot = plt.Circle(self.coord, 20)
    drawing.ax.add_artist(self.dot)
    low, high = drawing.crange
    self.amp = high-low
    self.low = low

  def update(self, data):
    self.txt.set_text(self.text % data[self.label])
    self.dot.set_color(cm.coolwarm((data[self.label] - self.low) / self.amp))


class Time(object):
  """To print the time of the experiment.

  It will print the time since the `t0` of the block.
  """

  def __init__(self, drawing, **kwargs):
    self.coord = kwargs['coord']

    self.txt = plt.text(self.coord[0], self.coord[1], "00:00", size=38)
    self.block = drawing

  def update(self, *_):
    self.txt.set_text(str(timedelta(seconds=int(time()-self.block.t0))))


elements = {'text': Text, 'dot_text': Dot_text, 'time': Time}

# ========== The block itself ==========


class Drawing(Block):
  """Block to make a visual representation of data."""

  def __init__(self,
               image,
               draw=None,
               crange=None,
               title="Drawing",
               window_size=(7, 5),
               freq=2,
               backend="TkAgg"):
    """Sets the args and initializes the parent block.

    Args:
      image: This image will be the background for the Canvas.
      draw (:obj:`dict`, optional): A :obj:`list` of :obj:`dict` defining what
        to draw. See below for more details.
      crange:
      title:
      window_size:
      freq:
      backend:

    Note:
      - ``draw`` keys:

        - ``type`` (:obj:`str`): Mandatory, the type of drawing to display. It
          can be either `'Text'`, `'Dot_text'` or `''Time`.

        - ``coord`` (:obj:`list`): Mandatory, a :obj:`list` containing the `x`
          and `y` coordinates where the drawing should be displayed.

        - ``text``: Mandatory for :class:`Text` and :class:`Dot_text` only, the
          left part of the displayed string.

        - ``label`` (:obj:`str`): Mandatory for :class:`Text` and
          :class:`Dot_text` only, the label of the data to display. It will be
          append to the ``text``.
    """

    Block.__init__(self)
    if draw is None:
      draw = []
    self.freq = freq
    self.image = image
    self.draw = draw
    self.crange = [20, 300] if crange is None else crange
    self.title = title
    self.window_size = window_size
    self.backend = backend

  def prepare(self):
    plt.switch_backend(self.backend)
    self.fig, self.ax = plt.subplots(figsize=self.window_size)
    image = self.ax.imshow(plt.imread(self.image), cmap=cm.coolwarm)
    image.set_clim(-0.5, 1)
    cbar = self.fig.colorbar(image, ticks=[-0.5, 1], fraction=0.061,
                             orientation='horizontal', pad=0.04)
    cbar.set_label('Temperatures(C)')
    cbar.ax.set_xticklabels(self.crange)
    self.ax.set_title(self.title)
    self.ax.set_axis_off()

    self.elements = []
    for d in self.draw:
      self.elements.append(elements[d['type']](self, **d))

  def loop(self):
    data = self.get_last()
    for elt in self.elements:
      elt.update(data)
    self.fig.canvas.draw()
    plt.pause(0.001)

  def finish(self):
    plt.close()
