# coding: utf-8
from _meta import MasterBlock
import time
#import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError
import os

class Streamer(MasterBlock):
	"""
Send a fake stream of data.
	"""
	def __init__(self,sensor=None,labels=['t(s)','signal']):
		"""
Use it for testing and LabJack streaming.

Parameters
----------
sensor: None or LabJack sensor instance
	If None, will stream an incremented integer.
	If Labjack sensor instance, will stream the LabJack Values as definied at 
	the instance creation. Be aware that if this block loops slowler than the 
	LabJack streams, it will crash the LabJack when buffer is full.
	If scansPerRead !=1, you can directly link output data to a graph/save without
	going through a compacter.
labels : list of str, default = ['t(s)','signal']
	Output labels.
		"""
		self.labels=labels
		self.sensor=sensor
		
	def main(self):
		self.i=0
		timer=time.time()
		print "Streamer : ", os.getpid()
		if self.sensor!=None:
			#self.sensor.close()
			self.sensor.new()
			time.sleep(1)
			print "device openend"
		try:
			while True:
				if self.sensor==None:
					time.sleep(0.001)
					try:
						for output in self.outputs:
							output.send(OrderedDict(zip(self.labels,[time.time()-self.t0,self.i])))
					except TimeoutError:
						raise
					except AttributeError: #if no outputs
						pass
					self.i+=1     
				else:
					self.freq=self.sensor.scanRate
					while time.time()-timer< 1./self.freq:
						time.sleep(1./(100*self.freq))
					timer=time.time()
					t,value=self.sensor.readStream()
					data=t-self.t0
					if self.sensor.scansPerRead!=1: # In case there is multiple values read on each steps
						value=[[value[i] for i in range(j,len(value),self.sensor.nchans)] for j in range(self.sensor.nchans)]
						value.insert(0,[data]*self.sensor.scansPerRead)
					else:
						value.insert(0,data)
					if self.labels==None:
						self.Labels=[i for i in range(self.sensor.nchans+1)]
					Array=OrderedDict(zip(self.labels,value))
					try:
						for output in self.outputs:
							output.send(Array)
					except TimeoutError:
						raise
					except AttributeError: #if no outputs
						pass
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in streamer: ",
			self.sensor.close()