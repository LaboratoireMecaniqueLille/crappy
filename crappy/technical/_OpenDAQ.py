from opendaq import DAQ
from time import time
from ._meta import io


class OpenDAQ(io.Control_Command):
    """
    Class for openDAQ Devices.
    """

    def __init__(self, *args, **kwargs):
        self.channels = kwargs.get('channels', 1)  # Possible values: 1..8
        self.gain = kwargs.get('gain', 1)  # Possible values: 0..4 (x1/3, x1, x2, x10, x100)
        self.offset = kwargs.get('offset', 0)  # not a parameter. apply after reading it.
        self.nsamples = kwargs.get('nsamples', 20)  # possible values : 0..254
        self.new()

    def new(self):
        self.handle = DAQ("/dev/ttyUSB0")
        self.handle.conf_adc(pinput=self.channels, ninput=0, gain=self.gain, nsamples=self.nsamples)
        pass

    def get_data(self, mock=None):
        data = self.handle.read_analog()
        return time(), [data]

    def close(self):
        self.handle.close()
        pass
