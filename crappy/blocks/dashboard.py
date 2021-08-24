# coding: utf-8

import threading
from queue import Queue
import numpy as np

from .block import Block
from .._global import OptionalModule

try:
  from tkinter import Tk, Label
except (ModuleNotFoundError, ImportError):
  Tk = OptionalModule("tkinter")
  Label = OptionalModule("tkinter")


class Dashboard(Block):
  """The Dashboard receives data from a :ref:`Link`, and prints it on a new
  popped window.

  It can receive either a single point, or a :obj:`list` of points. In this
  case, the displayed value corresponds to the average of points.
  """

  def __init__(self, labels=None, nb_digits=3):
    """Sets the args and initializes parent class.

    Args:
      labels (:obj:`list`, optional): Values to plot on the output window. If
        :obj:`None`, will plot every data.
      nb_digits (:obj:`int`, optional): Number of decimals to show.
    """

    super(Dashboard, self).__init__()
    self.labels = labels
    self.nb_display_values = len(self.labels) if self.labels else None
    self.nb_digits = nb_digits
    # global queue
    self.queue = Queue()

  class Dashboard:
    """Dashboard class created, is launched in a new thread."""

    def __init__(self, labels, nb_digits, queue):
      self.root = Tk()
      self.root.title('Dashboard')
      self.root.resizable(width=False, height=False)
      self.first_column = labels
      self.nb_digits = nb_digits
      self.c2 = []
      self.queue = queue
      # Creating the first and second column. Second column will be updated.
      for row_index, first_column in enumerate(self.first_column):
        Label(self.root, text=first_column, borderwidth=15,
              font=("Courier bold", 48)).grid(row=row_index, column=0)
        self.c2.append(
          Label(self.root, text='', borderwidth=15, font=("Courier bold", 48)))
        self.c2[row_index].grid(row=row_index, column=1)
      self.i = 0
      while True:
        self.update()

    def update(self):
      """Method to update the output window."""

      values = self.queue.get()
      for row, text in enumerate(values):
        self.c2[row].configure(text='%.{}f'.format(self.nb_digits) % text)
      self.root.update()

  def main(self):
    """Main loop."""

    if not self.labels:
      self.labels = list(self.inputs[0].recv(blocking=True).keys())
      self.nb_display_values = len(self.labels)
    dash_thread = threading.Thread(target=self.Dashboard,
                                   args=(self.labels, self.nb_digits,
                                         self.queue))
    dash_thread.daemon = True
    dash_thread.start()
    list_to_show = []
    while True:
      data_received = self.inputs[0].recv(blocking=True)
      if len(self.labels) == len(data_received):
        time = np.mean(list(data_received.values())[0])
        values = [np.mean(list(data_received.values())[label]) for label in
                  range(1, self.nb_display_values)]
        list_to_show.append(time)
        list_to_show.extend(values)
      else:
        for label in self.labels:
          list_to_show.append(
            np.around(np.mean(data_received[label]), self.nb_digits))
      self.queue.put(list_to_show)
      list_to_show = []
