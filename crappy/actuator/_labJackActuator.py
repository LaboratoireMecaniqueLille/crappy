# coding: utf-8
#from ._meta import acqSensor
#import numpy as np
#import time
import comedi as c
#from multiprocessing import Array
#import os
#import sys, string, struct
from labjack import ljm


class LabJackActuator(object):
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
		self.handle=ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, "ANY")
		
		
	def set_cmd(self,cmd):
		"""Convert the tension value to a digital value and send it to the output."""
		#self.out=(cmd/self.gain)-self.offset
		#out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the cmd to digital value
		self.out=(cmd*self.gain)+self.offset
		ljm.eWriteName(self.handle, self.channel, self.out)
		#print "sent : ", self.out
			
	def close(self):
		"""close the output."""
		#c.comedi_cancel(self.device,self.subdevice)
		ljm.close(handle)
		print "output handle closed"
	
      
  #def set_PID(self,wanted_position,sensor_input):
    #"""send a signal through a PID, based on the wanted command and the sensor_input"""
    #self.time= time.time()
    #self.out=(wanted_position/self.gain)-self.offset

    #self.error=self.out-sensor_input
    #self.I_term += self.Ki*self.error*(self.last_time-self.time)
    
    #if self.I_term>self.out_max:
      #self.I_term=self.out_max
    #elif self.I_term<self.out_min:
      #self.I_term=self.out_min
    
    #self.out_PID=self.last_output+self.K*self.error+self.I_term-self.Kd*(sensor_input-self.last_sensor_input)/(self.last_time-self.time)
    
    #if self.out_PID>self.out_max:
      #self.out_PID=self.out_max
    #elif self.out_PID<self.out_min:
      #self.out_PID=self.out_min
      
    #self.last_time=copy.copy(self.time)
    #self.last_sensor_input=copy.copy(sensor_input)
    #self.last_output=copy.copy(self.out_PID)
    #out_a=c.comedi_from_phys(self.out_PID,self.range_ds,self.maxdata) # convert the wanted_position 
    #c.comedi_data_write(self.device0,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
    #t=time.time()
    #return (t,self.out_PID)