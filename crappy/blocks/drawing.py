#coding: utf-8
from __future__ import print_function,division

from datetime import timedelta
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm

from .block import Block

# ======= Visual objects =========
# These classes represent all that can be drawn on the canvas
# Commmon arg:
#  drawing: The drawing itself. It is used to access its attributes when needed
# Common kwargs:
#   coord: The coordinates of the object on the Cavas
#
# Update method will be called frequently by the Drawing block, you can
# define here what it will do on each update
# It gives the argument "data" containing all the latest data received
# by the block


class Text(object):
  """
  A simple text line.

  Args:
    - text (constant): The left part of the displayed string.
    - label: The label to get the data to update.

      Note:
        It will be appended to the constant text.

  """
  def __init__(self,drawing,**kwargs):
    for k in ['coord','text','label']:
      setattr(self,k,kwargs[k])
    self.txt = plt.text(self.coord[0],self.coord[1],self.text)

  def update(self,data):
    self.txt.set_text(self.text%data[self.label])


class Dot_text(object):
  """
  Like Text, but with a colored dot to visualize a numerical value.

  Args:
    - *See Text*

  Warning!
    The value received in the label MUST be a numeric value.

  Note:
    It will be normalized on the crange of the block and the dot will change
    color from blue to red depending on this value.

  """
  def __init__(self,drawing,**kwargs):
    for k in ['coord','text','label']:
      setattr(self,k,kwargs[k])
    self.txt = plt.text(self.coord[0]+40,self.coord[1]+20,self.text,size=16)
    self.dot = plt.Circle(self.coord,20)
    drawing.ax.add_artist(self.dot)
    low,high = drawing.crange
    self.amp = high-low
    self.low = low

  def update(self,data):
    self.txt.set_text(self.text%data[self.label])
    self.dot.set_color(cm.coolwarm((data[self.label]-self.low)/self.amp))


class Time(object):
  """
  To print the time of the experiment.

  Args:
    - *None in particular*

  Note:
    It will print the time since the t0 of the block.

  """
  def __init__(self,drawing,**kwargs):
    for k in ['coord']:
      setattr(self,k,kwargs[k])
    self.txt = plt.text(self.coord[0],self.coord[1],"00:00",size=38)
    self.block = drawing

  def update(self,data):
    self.txt.set_text(str(timedelta(seconds=int(time()-self.block.t0))))


elements = {'text':Text,'dot_text':Dot_text,'time':Time}

# ========== The block itself ==========


class Drawing(Block):
  """
  Block to make a visual representation of data.

  Args:
    - image: The only mandatory argument.

      Note:
        This image will be the background for the Canvas.

    - draw: A list of dict defining what to draw.

      Warning!
        Each dict must contain a 'type' key that contains the name of the
        element, Drawing will then create the corresponding class with all the
        other keys as argument.

  """
  def __init__(self,image,draw=[],crange=[20,300],title="Drawing",
      window_size=(7,5),freq=2,backend="TkAgg"):
    Block.__init__(self)
    self.freq = freq
    self.image = image
    self.draw = draw
    self.crange = crange
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
      self.elements.append(elements[d['type']](self,**d))

  def loop(self):
    data = self.get_last()
    for elt in self.elements:
      elt.update(data)
    self.fig.canvas.draw()
    plt.pause(0.001)

  def finish(self):
    plt.close()
