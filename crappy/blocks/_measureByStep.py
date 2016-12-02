# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MeasureByStep MeasureByStep
# @{

## @file _measureByStep.py
# @brief Streams value measured on a card through a Link object.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _meta import MasterBlock
import time
import os
from collections import OrderedDict
from ..links._link import TimeoutError


class MeasureByStep(MasterBlock):
    """
    Streams value measured on a card through a Link object.
    """

    def __init__(self, sensor, labels=None, freq=None):
        """
        This streamer read the value on all channels ONE BY ONE and send the values through a Link object.

        It is slower than StreamerComedi, but works on every USB driver.
        It also works on LabJack devices.

        It can be triggered by a Link sending boolean (through "add_input" method),
        or internally by defining the frequency.

        Args:
            sensor:     sensor object
                        See sensor.sensor documentation. Tested for LabJackSensor and ComediSensor.
            labels:     list, optional
                        The labels you want on your output data. If None, will be time(sec) as first arg, and the
                        channel number as additional args.
            freq :      float or int, optional
                        Wanted acquisition frequency. If none, will be at the software looping speed.
        """
        super(MeasureByStep, self).__init__()
        self.sensor = sensor
        self.labels = labels if labels else ["time(sec)"] + self.sensor.channels
        self.freq = freq

    def main(self):
        """
        Main loop for MeasureByStep. Retrieves data at specified frequency (or software looping speed) from specified
        sensor, and sends it to a crappy link.
        """
        try:
            print "measureByStep / main loop: PID", os.getpid()
            trigger = "internal" if len(self.inputs) == 0 else "external"
            timer = time.time()
            while True:
                if trigger == "internal":
                    if self.freq:  # timing loop
                        while time.time() - timer < 1. / self.freq:
                            time.sleep(1. / (100 * self.freq))
                        timer = time.time()
                    sensor_epoch, sensor_values = self.sensor.get_data("all")
                    chronometer = sensor_epoch - self.t0
                    sensor_values.insert(0, chronometer)
                if trigger == "external":
                    if self.inputs[0].recv():  # wait for a signal
                        pass
                    sensor_epoch, sensor_values = self.sensor.get_data("all")
                    chronometer = sensor_epoch - self.t0
                    sensor_values.insert(0, chronometer)

                results = OrderedDict(zip(self.labels, sensor_values))
                try:
                    for output in self.outputs:
                        output.send(results)
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass

        except (Exception, KeyboardInterrupt) as e:
            print "Exception in measureByStep : ", e
            self.sensor.close()
