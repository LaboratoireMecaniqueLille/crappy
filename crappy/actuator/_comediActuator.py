# coding: utf-8
#from ._meta import acqSensor
#import numpy as np
#import time
from ._meta import command
import comedi as c
#from multiprocessing import Array
#import os
#import sys, string, struct


class ComediActuator(command.Command):
    """Comedi actuator object, commands the output of comedi cards"""
    def __init__(self,device='/dev/comedi0',subdevice=1,channel=0,range_num=0,gain=1,offset=0): 
        """Convert wanted tension value into digital values and send it to the 
        output of some Comedi-controlled card.
        
        Output is (command * gain) + offset.
        
        Parameters
        ----------
        device : str, default = '/dev/comedi0'
                Path to the device.
        subdevice : int, default = 1
                Subdevice 1 is the output.
        channel : int, default = 0
                The desired output channel.
        range_num : int, default = 0
                See the comedi documentation for different values.
        gain : float, default = 1
                Multiplication gain for the output.
        offset : float, default = 0
                Add this value to your output.
        """
        self.subdevice= subdevice
        self.channel=channel
        self.range_num=range_num
        self.gain=gain
        self.offset= offset
        self.device=c.comedi_open(device)
        self.new()
        
    def new(self):
        self.maxdata=c.comedi_get_maxdata(self.device,self.subdevice,self.channel)
        self.range_ds=c.comedi_get_range(self.device,self.subdevice,self.channel,self.range_num)
        c.comedi_dio_config(self.device,2,self.channel,1)

    def set_cmd(self,cmd):
        """
        Convert the tension value to a digital value and send it to the output.
        """
        self.out=(cmd*self.gain)+self.offset
        out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the cmd 
        c.comedi_data_write(self.device,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
    
    def On(self):
        c.comedi_dio_write(self.device,2,self.channel,1)
        
    def Off(self):
        c.comedi_dio_write(self.device,2,self.channel,0)
                
    def close(self):
        """
        close the output.
        """
        ret = c.comedi_close(self.device)
        if ret !=0: raise Exception('comedi_close failed...')