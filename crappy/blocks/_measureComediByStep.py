# coding: utf-8
from _meta import MasterBlock
import time
#import pandas as pd
import os
from collections import OrderedDict
from ..links._link import TimeoutError


class MeasureComediByStep(MasterBlock):
	"""
Streams value measure on a comedi card through a Link object.
	"""
	def __init__(self,comediSensor,labels=None,freq=None):
		"""
DEPRECATED : This block is to be replaced by MeasureByStep
This streamer read the value on all channels ONE BY ONE and send the 
values through a Link object. it is slower than StreamerComedi, but works on 
every USB driver. 

It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters
----------
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list
	The labels you want on your output data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition card capability.
		"""
		self.comediSensor=comediSensor
		self.labels=labels
		self.freq=freq
		print "DEPRECATED : Please use the MeasureByStep block"

	def main(self):
		try:
			try:
				print "measurecomedi : ", os.getpid()
				_a=self.inputs[:]
				trigger="external"
			except AttributeError:
				trigger="internal"
			timer=time.time()
			while True:
				if trigger=="internal":
					if self.freq!=None:
						while time.time()-timer< 1./self.freq:
							time.sleep(1./(100*self.freq))
						timer=time.time()
					data=[time.time()-self.t0]
					for channel_number in range(self.comediSensor.nchans):
						t,value=self.comediSensor.getData(channel_number)
						data.append(value)
				if trigger=="external":
					if self.inputs.input_.recv(): # wait for a signal
						data=[time.time()-self.t0]
					for channel_number in range(self.comediSensor.nchans):
						t,value=self.comediSensor.getData(channel_number)
						data.append(value)
				if self.labels==None:
					self.Labels=[i for i in range(self.comediSensor.nchans+1)]
				#Array=pd.DataFrame([data],columns=self.labels)
				#print data, self.labels
				Array=OrderedDict(zip(self.labels,data))
				try:
					for output in self.outputs:
						output.send(Array)
				except TimeoutError:
					raise
				except AttributeError: #if no outputs
					pass

		except (Exception,KeyboardInterrupt) as e:
			print "Exception in measureComediByStep : ", e
			self.comediSensor.close()
			#raise
