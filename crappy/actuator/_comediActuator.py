#from ._meta import acqSensor
import numpy as np
import time
import comedi as c
#from multiprocessing import Array
#import os
#import sys, string, struct



class ComediActuator(object): #acqSensor.AcqSensor
	"""
	Comedi actuator object, commands the output of comedi cards
	"""
	def __init__(self,device='/dev/comedi0',subdevice=1,channel=0,range_num=0,gain=1,offset=0): 
		self.subdevice=subdevice
		self.channel=channel
		self.range_num=range_num
		self.gain=gain
		self.offset=offset
		self.device=c.comedi_open(device)
		self.maxdata=c.comedi_get_maxdata(self.device,self.subdevice,self.channel)
		self.range_ds=c.comedi_get_range(self.device,self.subdevice,self.channel,self.range_num)
		
		
	def set_cmd(self,cmd):
		"""send a signal"""
		self.out=(cmd/self.gain)-self.offset
		out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the cmd to digital value
		self.out=(cmd*self.gain)+self.offset
		out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the cmd 
		c.comedi_data_write(self.device,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
		#t=time.time()
		#return (t,self.out)

			
	def close(self):
		#c.comedi_cancel(self.device,self.subdevice)
		ret = c.comedi_close(self.device)
		if ret !=0: raise Exception('comedi_close failed...')
	
      
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