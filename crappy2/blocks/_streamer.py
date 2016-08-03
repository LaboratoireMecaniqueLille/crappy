# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Streamer Streamer
# @{

## @file _streamer.py
# @brief Send a stream of data.
# @author Corentin Martel
# @version 0.1
# @date 13/07/2016

from _meta import MasterBlock
import time
# import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError
import os


class Streamer(MasterBlock):
    """
    Send a stream of data.
    """

    def __init__(self, sensor=None, labels=None):
        """
        Use it for testing and LabJack streaming.

        Args:
            sensor: LabJack sensor instance
                Will stream the LabJack Values as definied at
                the instance creation. Be aware that if this block loops slower than the
                LabJack streams, it will crash the LabJack when buffer is full.
                You can directly link output data to a graph/save without
                going through a compacter.
            labels : list of str, default = name of used sensor channels
                Output labels.
        """
        super(Streamer, self).__init__()
        self.sensor = sensor
        self.freq = self.sensor.scanRate
        if labels:
            self.labels = labels
        else:
            self.labels = ['t(s)']
            for index_channel in range(len(self.sensor.labels)):
                self.labels.append(self.sensor.labels[index_channel])

    def main(self):
        print "Streamer : ", os.getpid()
        self.sensor.start_stream()
        self.freq = self.sensor.scansPerRead
        self.t0 = time.time()
        try:
            while True:
                # self.countdown = time.time()
                # while time.time() - self.countdown < 1. / self.freq:
                #     time.sleep(1. / (100 * self.freq))
                t, value = self.sensor.get_data()
                data = [t - self.t0]
                for index_freq in range(1, self.freq):
                    data.append(data[-1] + 1. / (self.sensor.scansPerRead * self.sensor.scanRate))
                value = (data, value)
                array = OrderedDict(zip(self.labels, value))
                try:
                    for output in self.outputs:
                        output.send(array)
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in streamer: ", e
            if not self.sensor:
                self.sensor.close()
