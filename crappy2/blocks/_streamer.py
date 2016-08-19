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
from os import getpid
from numpy import arange
from multiprocessing import Process, Queue


class Streamer(MasterBlock):
    """
    Send a stream of data.
    """
    def __init__(self, sensor=None, labels=None):
        """
        Use it for LabJack streaming.
        You can directly link output data to a graph/save without going through a compacter.
        Args:
            sensor: LabJack sensor instance
                    Will read the streaming buffer of the LabJack device as defined at
                    the instance creation.
            labels : list of str, default = name of used sensor channels output labels.

        PERFORMANCE CONSIDERATIONS

        This block does the time vector reconstruction, then assemble it with the results matrix read from the
        LJM Buffer. You have to make sure the computer you run the program on has enough computing power to create
        the time vector of length = scan_rate in less than 1 second. Otherwise the LJM_Buffer will fill endlessly...
        """
        super(Streamer, self).__init__()
        assert sensor, "No input sensor defined."
        self.sensor = sensor
        self.nb_samples = self.sensor.scans_per_read
        self.labels = labels if labels else ['t(s)'] + self.sensor.channels
        global queue
        queue = Queue(2)

    def time_vector(self):
        """
        Time vector reconstruction. This has to be made in another process. Creates the time vector of the
        length = nb_samples, then puts it in a queue to be retrieved as soon as the result matrix pops.
        """
        sample_number = 0
        print "Streamer / time vector reconstruction: PID", getpid()
        try:
            while True:
                ratio = sample_number * self.nb_samples
                time_vector = (arange(ratio, ratio + self.nb_samples) / float(self.sensor.scan_rate)).tolist()
                queue.put(time_vector)
                sample_number += 1
        except:
            while not queue.empty():  # flush the queue and ?close it
                queue.get_nowait()
            # queue.close()

    def main(self):
        """
        Main loop of the streamer program.
        """
        print "Streamer / main loop: PID", getpid()
        time_vector_process = Process(target=self.time_vector)
        try:
            self.sensor.start_stream()
            time_vector_process.start()
            while True:
                value = (queue.get(), self.sensor.get_stream()[1])
                array = OrderedDict(zip(self.labels, value))
                try:
                    for output in self.outputs:
                        output.send(array)
                        print "test1"
                except TimeoutError:
                    print "test2"
                    raise
                except AttributeError:  # if no outputs
                    print "test3"
                    raise
        except KeyboardInterrupt:
            pass
        except Exception:
            raise





