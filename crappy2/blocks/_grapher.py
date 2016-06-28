# coding: utf-8
from _meta import MasterBlock
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan', linewidth=500)
# import pandas as pd
import os
from collections import OrderedDict


class Grapher(MasterBlock):
    """Plot the input data"""

    def __init__(self, mode, *args):
        """
        The grapher receive data from the Compacter (via a Link) and plots it.

        Parameters
        ----------
        mode : {"dynamic","static"}
            * "dynamic" : create a dynamic graphe that updates in real time.

            * "static" : create a graphe that add new values at every refresh. If there \
            is too many data (> 20000), delete one out of 2 to avoid memory overflow.

        args : tuple
            tuples of the columns labels of input data for plotting. You can add as\
            much as you want, depending on your computer performances.

        Examples
        --------
            graph=Grapher("dynamic",('t(s)','F(N)'),('t(s)','def(%)'))
                plot a dynamic graph with two lines plot(F=f(t) and def=f(t)).
            graph=Grapher("static",('def(%)','F(N)'))
                plot a static graph.
        """
        super(Grapher, self).__init__()
        print "grapher!"
        self.mode = mode
        self.args = args
        self.nbr_graphs = len(args)

    def main(self):
        try:
            print "main grapher", os.getpid()
            if self.mode == "dynamic":
                # print "1"
                save_number = 0
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                for i in range(self.nbr_graphs):  # init lines
                    if i == 0:
                        li = ax.plot(np.arange(1), np.zeros(1))
                    else:
                        li.extend(ax.plot(np.arange(1), np.zeros(1)))
                plt.grid()
                # print "2"
                fig.canvas.draw()  # draw and show it
                plt.show(block=False)
                while True:
                    # print "3"
                    Data = self.inputs[0].recv()  # recv data

                    if type(Data) is not OrderedDict:
                        Data = OrderedDict(zip(Data.columns, Data.values[0]))
                    # legend_=Data.columns[1:]
                    legend_ = [self.args[i][1] for i in range(self.nbr_graphs)]
                    if save_number > 0:  # lose the first round of data
                        if save_number == 1:  # init
                            var = Data
                            # box = ax.get_position()
                            # ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            # box.width, box.height * 0.9])
                            # plt.legend(legend_,bbox_to_anchor=(0, -0.14, 1., .102),
                            # loc=3, ncol=len(legend_), mode="expand", borderaxespad=-2.)
                            plt.legend(legend_, bbox_to_anchor=(-0.03, 1.02, 1.06, .102),
                                       loc=3, ncol=len(legend_), mode="expand",
                                       borderaxespad=1)
                        elif save_number <= 10:  # stack values
                            # var=pd.concat([var,Data])
                            # var=OrderedDict(zip(var.keys(),zip(var.values(),Data.values())))
                            try:
                                var = OrderedDict(zip(var.keys(), [var.values()[t] + Data.values()[t] for t in
                                                                   range(len(var.keys()))]))
                            except TypeError:
                                var = OrderedDict(zip(var.keys(), [(var.values()[t],) + (Data.values()[t],) for t in
                                                                   range(len(var.keys()))]))
                                # print var
                        else:  # delete old value and add new ones
                            # try:
                            # pass
                            # var=pd.concat([var[np.shape(Data)[0]:],Data])
                            # except AttributeError:
                            # var=OrderedDict(zip(var.keys(),zip(tuple(np.asarray(var.values())[np.shape(Data.values())[1]:]),Data.values())))
                            var = OrderedDict(zip(var.keys(),
                                                  [var.values()[t][np.shape(Data.values())[1]:] + Data.values()[t] for t
                                                   in range(len(var.keys()))]))
                        for i in range(self.nbr_graphs):  # update lines
                            li[i].set_xdata(var[self.args[i][0]])
                            li[i].set_ydata(var[self.args[i][1]])
                    ax.relim()
                    # print "4"
                    ax.autoscale_view(True, True, True)
                    fig.canvas.draw()
                    plt.pause(0.001)
                    if save_number <= 10:
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
                    # legend_=Data.columns[1:]
                    legend_ = [self.args[i][1] for i in range(self.nbr_graphs)]
                    if first_round:  # init at first round
                        for i in range(self.nbr_graphs):
                            if i == 0:
                                # print Data,Data[self.args[i][0]]
                                li = ax.plot(
                                    Data[self.args[i][0]], Data[self.args[i][1]],
                                    label='line ' + str(i))
                            else:
                                li.extend(ax.plot(
                                    Data[self.args[i][0]], Data[self.args[i][1]],
                                    label='line ' + str(i)))
                                # box = ax.get_position()
                                # ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                # box.width, box.height * 0.9])
                                # plt.legend(legend_,bbox_to_anchor=(0, -0.14, 1., .102),
                                # loc=3, ncol=len(legend_), mode="expand", borderaxespad=0.)
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
        # raise
        finally:
            plt.close('all')
