# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MeasureComediByStep MeasureComediByStep
# @{

## @file _measureComediByStep.py
# @brief Streams value measure on a comedi card through a Link object.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _meta import MasterBlock
import time
import os
from collections import OrderedDict
from ..links._link import TimeoutError


class MeasureComediByStep(MasterBlock):
    """
    Streams value measure on a comedi card through a Link object.
    """

    def __init__(self, comediSensor, labels=None, freq=None):
        """
        DEPRECATED : This block is to be replaced by MeasureByStep.

        This streamer read the value on all channels ONE BY ONE and send the
        values through a Link object. it is slower than StreamerComedi, but works on
        every USB driver.

        It can be triggered by a Link sending boolean (through "add_input" method),
        or internally by defining the frequency.

        Args:
            comediSensor : comediSensor object
                See sensor.ComediSensor documentation.
            labels : list
                The labels you want on your output data.
            freq : float or int, optional
                Wanted acquisition frequency. Cannot exceed acquisition card capability.
        """
        super(MeasureComediByStep, self).__init__()
        self.comediSensor = comediSensor
        self.labels = labels
        self.freq = freq
        print "[MeasureComediByStep] DEPRECATED : Please use the MeasureByStep block"

    def main(self):
        try:
            print "measurecomedi : ", os.getpid()
            trigger = "internal" if len(self.inputs) == 0 else "external"
            timer = time.time()
            while True:
                if trigger == "internal":
                    if self.freq is not None:
                        while time.time() - timer < 1. / self.freq:
                            time.sleep(1. / (100 * self.freq))
                        timer = time.time()
                    data = [time.time() - self.t0]
                    for channel_number in range(self.comediSensor.nchans):
                        t, value = self.comediSensor.get_data(channel_number)
                        data.append(value)
                if trigger == "external":
                    if self.inputs[0].recv():  # wait for a signal
                        data = [time.time() - self.t0]
                    for channel_number in range(self.comediSensor.nchans):
                        t, value = self.comediSensor.get_data(channel_number)
                        data.append(value)
                if self.labels is None:
                    self.Labels = [i for i in range(self.comediSensor.nchans + 1)]
                # Array=pd.DataFrame([data],columns=self.labels)
                # print data, self.labels
                Array = OrderedDict(zip(self.labels, data))
                try:
                    for output in self.outputs:
                        output.send(Array)
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass

        except (Exception, KeyboardInterrupt) as e:
            print "Exception in measureComediByStep : ", e
            self.comediSensor.close()
            # raise
