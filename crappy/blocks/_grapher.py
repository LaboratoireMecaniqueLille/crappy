# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Grapher Grapher
# @{

## @file _grapher.py
# @brief The grapher plots data received from the Compacter (via a Link).
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _masterblock import MasterBlock
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan', linewidth=500)
# import pandas as pd
import os
from collections import OrderedDict


class Grapher(MasterBlock):
  """Plot the input data"""

  def __init__(self, *args, **kwargs):
    """
    The grapher receive data from the Compacter (via a Link) and plots it.

    Args:
        args : tuple
            tuples of the columns labels of input data for plotting. You can add as\
            much as you want, depending on your computer performances.

        optional: length=x (int)
            number of chunks to data to be kept on the graph (default: 10)

            length=0 will create a static graph:
            add new values at every refresh. If there \
            is too many data (> 20000), delete one out of 2 to avoid memory overflow.

        optional: window_pos (tuple: (x, y) in PIXELS.)
            Window position when poping. The origin is in top left corner of screen.
            For instance, on a 1920x1080 screen, to have the window:
                in the top left corner: window_pos = (0, 0)
                in the top right corner: window_pos = (1920, 0)
                in the bottom left corner: window_pos  = (0, 1080)
            Be aware of the fact that a dual screen is considered as a surface with double the number of pixels, so
            the area of screen is (1920 + 1920, 1080 + 1080) for 2 screens of 1920x1080 definition.
        optional: window_size (tuple: (width, height), in INCHES).
            Self explanatory ?

    Examples
    --------
        graph=Grapher(('t(s)','F(N)'),('t(s)','def(%)'))
            plot a dynamic graph with two lines plot(F=f(t) and def=f(t)).
        graph=Grapher(('def(%)','F(N)'),length=0)
            plot a static graph.
        graph=Grapher(('t(s)','F(N)'),length=30)
            plot a dynamic graph that will display the last 30 chunks of data sent by the compacter
    """
    super(Grapher, self).__init__()
    self.len_graph = kwargs.get("length", 10)
    self.window_pos = kwargs.get("window_pos")
    self.window_size = kwargs.get("window_size", (8, 8))
    self.mode = "dynamic" if self.len_graph > 0 else "static"
    if isinstance(args[0], list):
      self.args = args[0]
    else:
      self.args = args
    self.nbr_graphs = len(self.args)

  def main(self):
    try:
      if self.mode == "dynamic":
        save_number = 0
        fig = plt.figure(figsize=self.window_size)
        ax = fig.add_subplot(111)
        for i in range(self.nbr_graphs):  # init lines
          if i == 0:
            li = ax.plot(np.arange(1), np.zeros(1))
          else:
            li.extend(ax.plot(np.arange(1), np.zeros(1)))
        plt.grid()
        if self.window_pos:
          mng = plt.get_current_fig_manager()
          mng.window.wm_geometry("+%s+%s" % self.window_pos)
        fig.canvas.draw()  # draw and show it
        plt.show(block=False)
        while True:
          Data = self.inputs[0].recv()  # recv data
          if type(Data) is not OrderedDict:
            Data = OrderedDict(zip(Data.columns, Data.values[0]))
          if type(Data[Data.keys()[0]]) != list:
            #got uncompacted data
            for k in Data:
              Data[k] = [Data[k]]
          legend_ = [self.args[i][1] for i in range(self.nbr_graphs)]
          if save_number > 0:  # lose the first round of data
            if save_number == 1:  # init
              var = Data
              plt.legend(legend_, bbox_to_anchor=(-0.03, 1.02, 1.06, .102),
                         loc=3, ncol=len(legend_), mode="expand",
                         borderaxespad=1)
            elif save_number <= self.len_graph:  # stack values
              try:
                var = OrderedDict(zip(var.keys(), [var.values()[t] + Data.values()[t] for t in
                                                   range(len(var.keys()))]))
              except TypeError:
                var = OrderedDict(zip(var.keys(), [(var.values()[t],) + (Data.values()[t],) for t in
                                                   range(len(var.keys()))]))
            else:  # delete old value and add new ones
              var = OrderedDict(zip(var.keys(),
                                    [var.values()[t][np.shape(Data.values())[1]:] + Data.values()[t] for t
                                     in range(len(var.keys()))]))
            for i in range(self.nbr_graphs):  # update lines
              li[i].set_xdata(var[self.args[i][0]])
              li[i].set_ydata(var[self.args[i][1]])
          ax.relim()
          ax.autoscale_view(True, True, True)
          fig.canvas.draw()
          plt.pause(0.001)
          if save_number <= self.len_graph:
            save_number += 1

      if self.mode == "static":
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        first_round = True
        k = [0] * self.nbr_graphs  # internal value for downsampling
        while True:
          Data = self.inputs[0].recv()  # recv data
          if type(Data) is not OrderedDict:
            Data = OrderedDict(zip(Data.columns, Data.values[0]))
          legend_ = [self.args[i][1] for i in range(self.nbr_graphs)]
          if first_round:  # init at first round
            for i in range(self.nbr_graphs):
              if i == 0:
                li = ax.plot(
                  Data[self.args[i][0]], Data[self.args[i][1]],
                  label='line ' + str(i))
              else:
                li.extend(ax.plot(
                  Data[self.args[i][0]], Data[self.args[i][1]],
                  label='line ' + str(i)))
            plt.legend(legend_, bbox_to_anchor=(-0.03, 1.02, 1.06, .102),
                       loc=3, ncol=len(legend_), mode="expand",
                       borderaxespad=1.)
            plt.grid()
            fig.canvas.draw()
            first_round = False
          else:  # not first round anymore
            for i in range(self.nbr_graphs):
              data_x = li[i].get_xdata()
              data_y = li[i].get_ydata()
              if len(data_x) >= 20000:
                # if more than 20000 values, cut half
                k[i] += 1
                li[i].set_xdata(np.hstack((data_x[::2],
                                           Data[self.args[i][0]][::2 ** k[i]])))
                li[i].set_ydata(np.hstack((data_y[::2],
                                           Data[self.args[i][1]][::2 ** k[i]])))
              else:
                li[i].set_xdata(np.hstack((data_x,
                                           Data[self.args[i][0]][::2 ** k[i]])))
                li[i].set_ydata(np.hstack((data_y,
                                           Data[self.args[i][1]][::2 ** k[i]])))
          ax.relim()
          ax.autoscale_view(True, True, True)
          fig.canvas.draw()

    except (Exception, KeyboardInterrupt) as e:
      print "Exception in grapher %s: %s" % (os.getpid(), e)
      plt.close('all')
      raise
