# coding: utf-8
#import numpy as np
import time
import comedi as c
from ._meta import acquisition
from .._deprecated  import _deprecated as deprecated
#from multiprocessing import Array
#import os
#import sys, string, struct



class ComediSensor(acquisition.Acquisition):
    """Sensor class for Comedi devices."""
    def __init__(self,device='/dev/comedi0',subdevice=0,channels=0,
                        range_num=0,gain=1,offset=0):
        """
        Convert tension value into digital values, on several channels.

        Output is (measured_value * gain) + offset.

        Parameters
        ----------
        device : str, default = '/dev/comedi0'
        Path to the device.
        subdevice : int, default = 0
        Subdevice 0 is the intput.
        channel : int or list of int, default = 0
        The desired output channel(s).
        range_num : int, default = 0
        See the comedi documentation for different values.
        gain : float or list of float, default = 1
        Multiplication gain for each channel. If there is multiple channels
        for a single gain, it will be applied to all.
        offset : float, default = 0
        Add this value for each channel. If there is multiple channels
        for a single offset, it will be applied to all.
        """
        self.subdevice=subdevice
        self.channels=channels
        self.range_num=range_num
        self.gain=gain
        self.offset=offset
        self.device=c.comedi_open(device)
        #if type(self.channels)==int or len(self.channels)==1:	# for getData
                #self.nchans=1
        if type(self.channels)==list:	# if multiple channels
                self.nchans=len(self.channels)
                self.range_num=[self.range_num]*self.nchans
                if type(self.gain)==int:
                        self.gain=[self.gain]*self.nchans
                if type(self.offset)==int:
                        self.offset=[self.offset]*self.nchans
                self.new()
        else:
                raise Exception("channels must be int or list")
    
                        
    def getData(self,channel_number="all"):
        """
        Read the signal for desired channel
        """
        if channel_number=="all":
            result=[]
            for channel in range(self.nchans):
                data = c.comedi_data_read(self.device,self.subdevice, self.channels[channel],                                   
                                          self.range_num[channel], c.AREF_GROUND)
                self.position=(c.comedi_to_phys(data[1],self.range_ds[channel],     
                                                    self.maxdata[channel])*self.gain[channel]+self.offset[channel])
                result.append(self.position)
            t=time.time()
            return (t, result)
        else:
            data = c.comedi_data_read(self.device,self.subdevice,
                                            self.channels[channel_number],
                                            self.range_num[channel_number], c.AREF_GROUND)
            self.position=(c.comedi_to_phys(data[1],self.range_ds[channel_number],
                                    self.maxdata[channel_number])*self.gain[channel_number]+self.offset[channel_number])
            t=time.time()
            return (t, self.position)
        
        
    def new(self):
        """ 
        Gather range and maxdata for all specified channels. This is only 
        called on init.
        """
        self.maxdata=[0]*self.nchans
        self.range_ds=[0]*self.nchans
        for i in range(self.nchans):
                self.maxdata[i]=c.comedi_get_maxdata(self.device,self.subdevice,
                                                                        self.channels[i])
                self.range_ds[i]=c.comedi_get_range(self.device,self.subdevice,
                                                                    self.channels[i],self.range_num[i])
    
    def close(self):
        """
        Close the device.
        """
        c.comedi_cancel(self.device,self.subdevice)
        ret = c.comedi_close(self.device)
        if ret !=0: raise Exception('comedi_close failed...')