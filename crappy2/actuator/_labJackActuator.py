# coding: utf-8
# from ._meta import acqSensor
# import numpy as np
# import time
# from multiprocessing import Array
# import os
# import sys, string, struct
from ._meta import command
from labjack import ljm
from .._warnings import deprecated as deprecated


class LabJackActuator(command.Command):
    """LabJack actuator object, commands the output of LabJack cards"""

    def __init__(self, channel="DAC0", gain=1, offset=0):
        """Convert wanted tension value into digital values and send it to the 
        output of some LabJack card.
        
        Output is (command * gain) + offset.
        
        Parameters
        ----------
        channel : str, default = "TDAC2"
                The desired output channel. See LabJack doc for possibilities.
        gain : float, default = 1
                Multiplication gain for the output.
        offset : float, default = 0
                Add this value to your output.
        """
        super(LabJackActuator, self).__init__()
        self.channel = channel
        self.gain = gain
        self.offset = offset
        self.new()

    def new(self):
        self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, "ANY")

    @deprecated(new)
    def open_handle(self):
        """
        DEPRECATED: The handle is now initialized in the new method.
        """
        self.new()

    def set_cmd(self, cmd):
        """
        Convert the tension value to a digital value and send it to the output.
        """
        out = (cmd * self.gain) + self.offset
        #print out
        ljm.eWriteName(self.handle, self.channel, out)
    
    def set_cmd_ram(self, cmd, address):
	data_type = ljm.constants.FLOAT32
	ljm.eWriteAddress(self.handle, address, data_type, cmd)
    
    def close(self):
        """
        close the output.
        """
        ljm.close(self.handle)
        print "output handle closed"
