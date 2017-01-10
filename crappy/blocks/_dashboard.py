# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Dashboard Dashboard
# @{

## @file _dashboard.py
# @brief The dashboard shows actual values of data received from the Compacter (via a Link).
#
# @author Francois Bari
# @version 0.1
# @date 10/01/2017

from _masterblock import MasterBlock
import os
from Tkinter import Tk, Label
import threading
from Queue import Queue
import numpy as np
import time

class Dashboard(MasterBlock):

  def __init__(self, *args, **kwargs):
    super(Dashboard, self).__init__()
    self.labels = kwargs.get('labels', None)
    self.nb_display_values = len(self.labels) if self.labels else None
    self.nb_digits = kwargs.get('nb_digits', 3)
    global queue
    queue = Queue()

  class Dashboard:
    def __init__(self, labels, nb_digits):
      self.root = Tk()
      self.root.title('Dashboard')
      self.root.resizable(width=False, height=False)
      self.first_column = labels
      self.nb_digits = nb_digits
      self.c2 = []
      # Creating the first and second column. Second column will be updated.
      for row_index, first_column in enumerate(self.first_column):
        Label(self.root, text=first_column, borderwidth=15, font=("Courier bold", 48)).grid(row=row_index, column=0)
        self.c2.append(Label(self.root, text='', borderwidth=15, font=("Courier bold", 48)))
        self.c2[row_index].grid(row=row_index, column=1)
      self.update()

    def update(self):
      values = queue.get()
      for row, text in enumerate(values):
        self.c2[row].configure(text='%.{}f'.format(self.nb_digits) % text)
      self.root.update()
      self.root.after(10, func=self.update())

  def main(self):
    if not self.labels:
      self.labels = self.inputs[0].recv(blocking=True).keys()
      self.nb_display_values = len(self.labels)
    dash_thread = threading.Thread(target=self.Dashboard, args=[self.labels, self.nb_digits])
    dash_thread.daemon = True
    dash_thread.start()
    list_to_show = []
    while True:
      try:
        data_received = self.inputs[0].recv(blocking=True)
        if len(self.labels) == len(data_received):
          time = np.mean(data_received.values()[0])
          values = [np.mean(data_received.values()[label]) for label in xrange(1, self.nb_display_values)]
          list_to_show.append(time)
          list_to_show.extend(values)
        else:
          for label in self.labels:
            list_to_show.append(np.around(np.mean(data_received[label]), self.nb_digits))
        queue.put(list_to_show)
        list_to_show = []

      except (Exception, KeyboardInterrupt) as e:
        print "Exception in dashboard %s: %s" % (os.getpid(), e)
        break
