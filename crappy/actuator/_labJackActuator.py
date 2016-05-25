# coding: utf-8
#from ._meta import acqSensor
#import numpy as np
#import time
#from multiprocessing import Array
#import os
#import sys, string, struct
from ._meta import command
from labjack import ljm
from .._deprecated  import _deprecated as deprecated

class LabJackActuator(command.Command):
    """LabJack actuator object, commands the output of LabJack cards"""
    def __init__(self,channel="TDAC2",gain=1,offset=0): 
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
        self.channel=channel
        self.gain=gain
        self.offset=offset
        self.open_handle()
    
    def new(self):
        self.handle=ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, "ANY")
        
    @deprecated(new)
    def open_handle(self):
        """
        DEPRECATED: The handle is now initialized in the new method.
        """
        self.new()
        
    def set_cmd(self,cmd):
        """
        Convert the tension value to a digital value and send it to the output.
        """
        self.out=(cmd*self.gain)+self.offset
        ljm.eWriteName(self.handle, self.channel, self.out)
                
                
    def close(self):
        """
        close the output.
        """
        ljm.close(self.handle)
        print "output handle closed"