# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Grapher Grapher
# @{

## @file grapher.py
# @brief The grapher plots data received from a block (via a Link).
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016


from .masterblock import MasterBlock
# Major library imports

import numpy as np
# Enthought imports
from traits.api import Array, HasTraits, Instance, Int
from traitsui.api import Item, View
from pyface.timer.api import Timer

# Chaco imports
from chaco.chaco_plot_editor import ChacoPlotItem
from threading import Thread
from Queue import Queue

queue = Queue()


class Plotter(HasTraits):
  x_data = Array
  y_data = Array
  length = Int(20)
  tick = Int(0)
  printed_points = Int(0)
  view = View(ChacoPlotItem("x_data", "y_data",
                            resizable=True,
                            x_label="Time",
                            y_label="Signal",
                            color="blue",
                            bgcolor="white",
                            border_visible=True,
                            border_width=1,
                            padding_bg_color="lightgray",
                            width=800,
                            height=380,
                            marker_size=2,
                            show_label=False),
              Item('length'),
              buttons=['OK'],
              resizable=True,
              width=800, height=500)

  def update_graph(self):
    print('je rentre dans update graph')
    recv = queue.get()
    self.nb_points = len(recv['time(sec)'])
    if self.tick == self.length:
      cur_index = self.x_data[-self.length * self.nb_points:]
      cur_data = self.y_data[-self.length * self.nb_points:]
    else:
      cur_index = self.x_data
      cur_data = self.y_data
      self.tick += 1

    new_y_data = np.hstack((cur_data, recv['signal']))
    new_x_data = np.hstack((cur_index, recv['time(sec)']))
    self.printed_points = len(new_x_data)
    self.x_data = new_x_data
    self.y_data = new_y_data
    return

  def _length_changed(self):
    print'NEW LENGTH!', self.length
    if self.length < self.printed_points:
      self.tick = 0
      self.x_data = self.x_data[-self.length * self.nb_points:]
      self.y_data = self.y_data[-self.length * self.nb_points:]
    else:
      pass


class Viewer(HasTraits):
  plotter = Instance(Plotter, ())
  view = View(  # Item('controller', style='custom', show_label=False),
    Item('plotter', style='custom', show_label=False),
    resizable=True, title='hello world')

  def configure_traits(self, *args, **kws):
    # self.timer = Timer(10, self.plotter.update_graph)
    return super(Viewer, self).configure_traits(*args, **kws)


class Rapher(MasterBlock, HasTraits):
  plotter = Instance(Plotter, ())
  view = View(  # Item('controller', style='custom', show_label=False),
    Item('plotter', style='custom', show_label=False),
    resizable=True, title='hello world')

  global queue

  def __init__(self, *args, **kwargs):
    super(Rapher, self).__init__()

  def main(self):
    thread = Thread(target=init_thread)
    thread.daemon = True
    thread.start()
    Timer(100, self.plotter.update_graph)

    while True:
      data = self.inputs[0].recv()
      queue.put(data)
      print('filling la queue')


def init_thread():
  viewer = Viewer()
  viewer.configure_traits()
