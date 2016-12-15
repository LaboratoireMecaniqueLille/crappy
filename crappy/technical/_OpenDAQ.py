from opendaq import DAQ
from time import time
from ._meta import io


class OpenDAQ(io.Control_Command):
    """
    Class for openDAQ Devices.
    """

    def __init__(self, *args, **kwargs):
        self.input_channels = kwargs.get('channels', 1)  # Possible values: 1..8
        self.input_gain = kwargs.get('gain', 1)  # Possible values: 0..4 (x1/3, x1, x2, x10, x100)
        self.input_offset = kwargs.get('offset', 0)  # not a parameter. apply after reading it.
        self.input_nsamples_per_read = kwargs.get('nsamples', 20)  # possible values : 0..254
        self.mode = kwargs.get('mode', 'single')
        self.new()

    def new(self):
        self.handle = DAQ("/dev/ttyUSB0")
        self.handle.conf_adc(pinput=self.input_channels, ninput=0, gain=self.input_gain, nsamples=self.input_nsamples_per_read)
        pass

    def get_data(self, mock=None):
        data = self.handle.read_analog()
        return time(), [data]

    def set_cmd(self, command):
        self.handle.set_dac(command)

    def close(self):
        self.handle.close()
        pass
