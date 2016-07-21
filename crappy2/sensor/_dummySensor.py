# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup dummysensor DummySensor
# @{

## @file _dummySensor.py
# @brief  Mock a sensor and return the time. Use it for testing.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016
import random
import time

from crappy2.sensor._meta import motion
from ._meta import acquisition


class DummySensor(acquisition.Acquisition, motion.MotionSensor):
    """Mock a sensor and return the time. Use it for testing."""

    def __init__(self, *args, **kwargs):
        super(DummySensor, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.position = random.random() * random.random() * 10
        if "channels" in kwargs:
            self.channels = kwargs.get("channels")
            if not isinstance(self.channels, list):
                self.channels = [self.channels]
        else:
            self.channels = [0]
        if "offset" in kwargs:
            self.offset = kwargs.get("offset")
            if not isinstance(self.offset, list):
                self.offset = [self.offset] * len(self.channels)
        else:
            self.offset = [0]
        if "gain" in kwargs:
            self.gain = kwargs.get("gain")
            if not isinstance(self.gain, list):
                self.gain = [self.gain] * len(self.channels)
        else:
            self.gain = [0]

    def get_data(self, channel_number=0):
        """
        Read the signal for desired channel
        """
        if channel_number == "all":
            result = []
            for i in range(len(self.channels)):
                result.append((round(random.random() * random.random() * 10, 3)) * self.gain[i] + self.offset[i])
            t = time.time()
            return t, result
        else:
            t = time.time()
            return t, round(random.random() * random.random() * 10, 3)

    def get_position(self):
        return self.position

    def new(self, *args):
        """Do nothing."""
        pass

    def close(self, *args):
        """Do nothing."""
        pass
