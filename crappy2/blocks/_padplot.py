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

import os
from collections import OrderedDict

np.set_printoptions(threshold='nan', linewidth=500)


class PadPlot(MasterBlock):
    """
    ONLY FOR TRIBOMETER AT THE MOMENT, but this block can be modified to draw everything on a figure.
    This block takes in input a dictionary containing measured values of thermocouples, then plots it in a drawing.

    Args:
        bg_image:           image to print in background of figure. Only prints a tribometer for the moment.
        colormap:           colormap to use for shapes and text.
        figure_title:       self-explanatory
        colormap_range:     extremas of colormap

    """

    def __init__(self, bg_image='./Pad2.png', colormap='coolwarm',
                 figure_title='Pad Temperature', colormap_range=[20, 30], *args, **kwargs):

        super(PadPlot, self).__init__()
        # 9 thermocouples, 9 circles
        self.coordinates_circles = (
            (110, 330), (92, 265), (68, 192), (507, 267), (507, 195), (507, 123), (900, 323), (920, 261), (940, 193), (160, 600), (760, 600))
        self.thermocouples_list = ['T' + str(count) for count in xrange(1, 10)] + ['Tdisc1', 'Tdisc2']# Names of each thermocouple
        self.coordinates_text_circles = (
            (150, 330), (132, 265), (108, 192), (547, 267), (547, 195), (547, 123), (940, 323), (960, 261), (980, 193), (200, 600), (800, 600))
        self.range = colormap_range
        self.bg_image = bg_image
        self.colormap = colormap
        self.figure_title = figure_title
        self.window_pos = kwargs.get("window_pos")
        self.window_size = kwargs.get("window_size")

    def get_data(self):
        """
        Get data from a link
        """
        data_full = self.inputs[0].recv()
        t = data_full.values()[0][0]
        data_ther = np.asarray(OrderedDict((k, data_full[k]) for k in self.thermocouples_list).values())
        return t, data_ther

    def normalize_thermocouples(self, data):
        """
        Normalization of thermocouple values for colomap call
        """
        temperature_min, temperature_max = float(self.range[0]), float(self.range[1])
        thermocouples_data_normalized = (data - temperature_min) / (temperature_max - temperature_min)
        # ratio_temperatures = temperature_min / temperature_max
        # temperature_min = data.min()
        # temperature_max = data.max()
        # thermocouples_data_normalized = (data - temperature_min) / (temperature_max - temperature_min)
        # ratio_temperatures = temperature_min / temperature_max
        return temperature_min, temperature_max, thermocouples_data_normalized

    def update_figure(self, data, time, time_elapsed_txt, minimum_txt, maximum_txt, circles, texts):
        temp_min, temp_max, temp_normalized = self.normalize_thermocouples(data)
        time_elapsed_txt.set_text('Time elapsed: %.1f sec' % time)
        minimum_txt.set_text('Global Tmin = %.1f' % temp_min)
        maximum_txt.set_text('Global Tmax = %.1f' % temp_max)

        for i in xrange(len(circles)):
            circles[i].set_color(cm.coolwarm(np.mean(temp_normalized[i])))
            texts[i].set_text(self.thermocouples_list[i] + '= %.1f' % np.mean(data[i]))
            texts[i].set_color(cm.coolwarm(np.mean(temp_normalized[i])))

    def main(self):
        print "Padplot / main loop: PID", os.getpid()
        try:
            fig, ax = plt.subplots(figsize=self.window_size)  # note we must use plt.subplots, not plt.subplot
            image = ax.imshow(plt.imread(self.bg_image), cmap='coolwarm')
            if self.window_pos:
                plt.get_current_fig_manager().window.wm_geometry("+%s+%s" % self.window_pos)
            image.set_clim(-0.5, 1)
            # mngr = plt.get_current_fig_manager()
            # mngr.window.wm_geometry(coord_window)
            ax.set_title(self.figure_title)
            ax.set_axis_off()
            # Now show the initial window
            circles = []
            texts = []
            for nb_circles in xrange(len(self.coordinates_circles)):
                circles.append(plt.Circle(self.coordinates_circles[nb_circles], 20))
                texts.append(
                    plt.text(self.coordinates_text_circles[nb_circles][0], self.coordinates_text_circles[nb_circles][1],
                             self.thermocouples_list[nb_circles] + '=' + str(0), size=16))

                ax.add_artist(circles[-1])
                texts[-1]  # to call it

            # now print some useful texts in specified locations
            minimum_txt = plt.text(50, 450, 'Global Tmin = %.1f' % 0., size=16)
            maximum_txt = plt.text(700, 450, 'Global Tmax = %.1f' % 0., size=16)
            time_elapsed_txt = plt.text(300, 750, 'Time elapsed: %.1f sec' % 0., size=20)

            ax.add_artist(minimum_txt)
            ax.add_artist(maximum_txt)
            ax.add_artist(time_elapsed_txt)
            fig.canvas.draw()
            fig.show()

            # list to update
            update_list = [time_elapsed_txt, minimum_txt, maximum_txt, circles, texts]
            while True:
                t, data = self.get_data()
                self.update_figure(data, t, *update_list)
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in PadPlot %s: %s" % (os.getpid(), e)
            plt.close('all')
        finally:
            plt.close('all')
