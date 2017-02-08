"""
Visualization of simulated live data stream

Shows how Chaco and Traits can be used to easily build a data
acquisition and visualization system.

Two frames are opened: one has the plot and allows configuration of
various plot properties, and one which simulates controls for the hardware
device from which the data is being acquired; in this case, it is a mockup
random number generator whose mean and standard deviation can be controlled
by the user.
"""
import time
# Major library imports
import numpy as np
from collections import OrderedDict

# Enthought imports
from traits.api import Array, HasTraits, Instance, Int
from traitsui.api import Item, View
from pyface.timer.api import Timer

# Chaco imports
from chaco.chaco_plot_editor import ChacoPlotItem
from multiprocessing import Process, Queue  # pour simuler le comportement

# de crappy

queue = Queue()


class Plotter(HasTraits):
  """
  Cette classe contient uniquement les points a afficher, et la methode qui
  permet de les mettre a jour.
  """
  # definition d'objets avec Traits
  x_data = Array
  y_data = Array
  length = Int(20)
  tick = Int(0)
  printed_points = Int(0)

  # pour positionner le grapher (le conteneur est la classe d'en dessous (
  # Viewer)
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
    """Methode appelee par le timer."""
    recv = queue.get()
    self.nb_points = len(recv['time(sec)'])
    if self.tick == self.length:  # pour creer un graph dynamique.
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
      self.x_data = self.x_data[-self.length:]
      self.y_data = self.y_data[-self.length:]
    else:
      pass


class Viewer(HasTraits):
  plotter = Instance(Plotter, ())
  view = View(  # Item('controller', style='custom', show_label=False),
    Item('plotter', style='custom', show_label=False),
    resizable=True, title='hello world')

  def configure_traits(self, *args, **kws):
    self.timer = Timer(1, self.plotter.update_graph)
    return super(Viewer, self).configure_traits(*args, **kws)


def create_vecteurs_de_ouf():
  """ pour simuler le comportement de crappy: je mets dans le lien 10 points
  pour chaque vecteur"""
  i = 0
  while True:
    timer = 0.1
    labels = ['time(sec)', 'signal']
    x = np.linspace(0 + i, 1 + i, 10, endpoint=False).tolist()
    y = np.sin(x).tolist()
    sent = OrderedDict([(labels[0], x), (labels[1], y)])
    queue.put(sent)
    time.sleep(timer)
    i += 1


p1 = Process(target=create_vecteurs_de_ouf)
p1.start()

plot = Viewer()
plot.configure_traits()  # pour demarrer le plot chaco

