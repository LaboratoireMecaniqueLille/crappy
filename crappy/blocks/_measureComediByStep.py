from _meta import MasterBlock
import time
#import pandas as pd
import os
from collections import OrderedDict

class MeasureComediByStep(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,comediSensor,labels=None,freq=None):
		"""
MeasureComediByStep(comediSensor,labels=None,freq=None)

This streamer read the value on all channels ONE BY ONE and send the 
values through a Link object. it is slower than StreamerComedi, but works on 
every USB driver. 
It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters:
-----------
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list
	The labels you want with your data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition card capability.
		"""
		self.comediSensor=comediSensor
		self.labels=labels
		self.freq=freq

	def main(self):
		try:
			print "measurecomedi : ", os.getpid()
			_a=self.inputs[:]
			trigger="external"
		except AttributeError:
			trigger="internal"
		timer=time.time()
		try:
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
				for output in self.outputs:
					output.send(Array)

		except (Exception,KeyboardInterrupt) as e:
			print "Exception in measureComediByStep : ", e
			self.comediSensor.close()
			raise
