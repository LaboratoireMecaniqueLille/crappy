# coding: utf-8
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
        This streamer read the value on all channels ONE BY ONE and send the
        values through a Link object. it is slower than StreamerComedi, but works on
        every USB driver.
        It also works on LabJack devices.

        It can be triggered by a Link sending boolean (through "add_input" method),
        or internally by defining the frequency.

        Parameters
        ----------
        sensor : sensor object
            See sensor.sensor documentation. Tested for LabJackSensor and ComediSensor.
        labels : list
            The labels you want on your output data.
        freq : float or int, optional
            Wanted acquisition frequency. Cannot exceed acquisition card capability.
        """
        super(MeasureByStep, self).__init__()
        self.sensor = sensor
        self.labels = labels
        self.freq = freq

    def main(self):
        try:
            try:
                print "measureByStep : ", os.getpid()
                _a = self.inputs[:]
                trigger = "external"
            except AttributeError:
                trigger = "internal"
            timer = time.time()
            while True:
                if trigger == "internal":
                    if self.freq != None:
                        while time.time() - timer < 1. / self.freq:
                            time.sleep(1. / (100 * self.freq))
                        timer = time.time()
                    # data=[time.time()-self.t0]
                    # for channel_number in range(self.sensor.nchans):
                    t, value = self.sensor.get_data("all")
                    data = t - self.t0
                    value.insert(0, data)
                if trigger == "external":
                    if self.inputs.input_.recv():  # wait for a signal
                        pass
                    # data=[time.time()-self.t0]
                    # for channel_number in range(self.sensor.nchans):
                    t, value = self.sensor.get_data("all")
                    data = t - self.t0
                    value.insert(0, data)
                if self.labels == None:
                    self.Labels = [i for i in range(self.sensor.nchans + 1)]
                # Array=pd.DataFrame([data],columns=self.labels)
                # print value, self.labels
                Array = OrderedDict(zip(self.labels, value))
                try:
                    for output in self.outputs:
                        output.send(Array)
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass

        except (Exception, KeyboardInterrupt) as e:
            print "Exception in measureComediByStep : ", e
            self.sensor.close()
        # raise
