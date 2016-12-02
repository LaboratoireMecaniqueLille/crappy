# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Grapher Grapher
# @{

## @file _padplt.py
# @brief The padplot receives data from thermocouples then plots it on a drawing.
#
# @author François Bari
# @version 0.1
# @date 05/09/2016

from _meta import MasterBlock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import timedelta

import os
from collections import OrderedDict

np.set_printoptions(threshold='nan', linewidth=500)


class CanvasDrawing(MasterBlock):
    """
    ONLY FOR TRIBOMETER AT THE MOMENT, but this block can be modified to draw everything on a figure.
    This block takes in input a dictionary containing measured values of thermocouples, then plots it in a drawing.

    Args:
        bg_image:           image to print in background of figure. Only prints a tribometer for the moment.
        colormap:           colormap to use for shapes and text.
        figure_title:       self-explanatory
        colormap_range:     extremas of colormap

    """

    def __init__(self, colormap='coolwarm', *args, **kwargs):

        super(CanvasDrawing, self).__init__()
        space_list = [0, 40]
        self.pad = {
            'T1': [(185 + space, 430) for space in space_list],  # T1
            'T2': [(145 + space, 320) for space in space_list],  # T2
            'T3': [(105 + space, 220) for space in space_list],  # T4
            'T4': [(720 + space, 370) for space in space_list],  # T5
            'T5': [(720 + space, 250) for space in space_list],  # T6
            'T6': [(720 + space, 125) for space in space_list],  # T7
            'T7': [(1220 + space, 410) for space in space_list],  # T8
            'T8': [(1260 + space, 320) for space in space_list],  # T9
            'T9': [(1300 + space, 230) for space in space_list],
            'T_disc': [(85 + space, 800) for space in space_list],
            'T_pad': [(85 + space, 880) for space in space_list],
        }
        self.thermocouples_list = self.pad.keys()
        print self.thermocouples_list
        # Optional parameters
        self.cmap_color = kwargs.get("cmap_color", 'coolwarm')
        self.bg_image = kwargs.get("bg_image")
        self.colormap_range = kwargs.get("colormap_range", [20, 100])
        self.figure_title = kwargs.get("figure_title", 'Canvas')
        self.window_pos = kwargs.get("window_pos")
        self.window_size = kwargs.get("window_size")

    def get_data(self):
        """
        Get data from a link, then returns the timeclock (scalar, first value) and a numpy array
        """
        data_full = self.inputs[0].recv()
        t = data_full.values()[0][0]
        data_thermocouple = np.asarray(OrderedDict((k, data_full[k]) for k in self.thermocouples_list).values())
        return t, data_thermocouple

    def normalize_thermocouples(self, data):
        """
        Normalization of thermocouple values for colomap call: each value of measured temperature will be between 0 and 1.
        """
        temperature_min, temperature_max = float(self.colormap_range[0]), float(self.colormap_range[1])
        thermocouples_data_normalized = (data - temperature_min) / (temperature_max - temperature_min)
        return temperature_min, temperature_max, thermocouples_data_normalized

    def update_figure(self, data, time, time_elapsed_txt, circles, texts):
        """
        Method to update the window with values read in data.

        Parameters
        ----------
        data: array containing values to update
        time: time to update on canvas
        time_elapsed_txt: class containing the text of time.
        circles : list of circle classes.
        texts : list of text classes

        """
        temp_min, temp_max, temp_normalized = self.normalize_thermocouples(data)
        time_elapsed_txt.set_text(str(timedelta(seconds=int(time))))

        for i in xrange(len(circles)):
            circles[i].set_color(cm.coolwarm(np.mean(temp_normalized[i])))
            texts[i].set_text(self.thermocouples_list[i] + '= %.1f' % np.mean(data[i]))
            texts[i].set_color(cm.coolwarm(np.mean(temp_normalized[i])))

    def main(self):
        print "Padplot / main loop: PID", os.getpid()
        try:
            fig, ax = plt.subplots(figsize=self.window_size)  # note we must use plt.subplots, not plt.subplot
            image = ax.imshow(plt.imread(self.bg_image), cmap=cm.coolwarm)
            image.set_clim(-0.5, 1)
            cbar = fig.colorbar(image, ticks=[-0.5, 1], fraction=0.061, orientation='horizontal', pad=0.04)
            cbar.set_label('Temperatures(C)')
            cbar.ax.set_xticklabels(self.colormap_range)
            if self.window_pos:
                plt.get_current_fig_manager().window.wm_geometry("+%s+%s" % self.window_pos)
            ax.set_title(self.figure_title)
            ax.set_axis_off()

            circles = []
            texts = []

            for key, value in self.pad.iteritems():
                circles.append(plt.Circle(value[0], 20))
                texts.append(plt.text(value[1][0], value[1][1], key + '=' + str(0), size=16))
                ax.add_artist(circles[-1])
                texts[-1]  # to call it

            # Now print some useful texts in specified locations
            time_elapsed_txt = plt.text(80, 1000, str(timedelta(seconds=0.0)), size=20)
            ax.add_artist(time_elapsed_txt)
            # Now show the initial window
            fig.canvas.draw()
            plt.show(block=False)

            # list to update
            update_list = [time_elapsed_txt, circles, texts]
            while True:
                t, data = self.get_data()
                self.update_figure(data, t, *update_list)
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()
                plt.pause(0.001)
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in PadPlot %s: %s" % (os.getpid(), e)
            plt.close('all')
        finally:
            plt.close('all')
