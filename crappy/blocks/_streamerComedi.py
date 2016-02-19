# coding: utf-8
from _meta import MasterBlock
import os
import numpy as np
import time
import struct
np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError


class StreamerComedi(MasterBlock):
	"""
Stream comedi values at high frequency.
	"""
	def __init__(self,comediSensor,labels=None,freq=8000,buffsize=10000):
		"""
WARNING :DOES NOT WORK AT THE MOMENT, USE measureComediByStep INSTEAD.

This streamer read the value on all channels at the same time and send the 
values through a Link object. It can be very fast, but needs need an USB 2.0
port drove by ehci-driver to work properly. 
xhci driver DOES NOT work (for now).

Parameters
----------
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list of str
	The labels you want with your data.
freq : int (default = 8000)
	the frequency you need.
buffsize : int, default = 10000
	Only use for testing purpose.
		"""
		import comedi as c
		self.labels=labels
		self.c=c
		self.comediSensor=comediSensor
		
		self.fd = self.c.comedi_fileno(self.comediSensor.device)	# get a file-descriptor

		self.BUFSZ = buffsize	# buffer size
		self.freq=freq	# acquisition frequency
	
		self.nchans = len(self.comediSensor.channels)	# number of channels
		self.aref =[self.c.AREF_GROUND]*self.nchans

		mylist = self.c.chanlist(self.nchans)	# create a chanlist of length nchans
		self.maxdata=[0]*(self.nchans)
		self.range_ds=[0]*(self.nchans)
		#print "1"
		for index in range(self.nchans):	# pack informations into the chanlist
			mylist[index]=self.c.cr_pack(self.comediSensor.channels[index],
						   self.comediSensor.range_num[index],
						   self.aref[index])
			self.maxdata[index]=self.c.comedi_get_maxdata(self.comediSensor.device,
									   self.comediSensor.subdevice,
									   self.comediSensor.channels[index])
			self.range_ds[index]=self.c.comedi_get_range(self.comediSensor.device,
									  self.comediSensor.subdevice,
									  self.comediSensor.channels[index],
									  self.comediSensor.range_num[index])

		cmd = self.c.comedi_cmd_struct()

		period = int(1.0e9/self.freq)	# in nanoseconds
		ret = self.c.comedi_get_cmd_generic_timed(self.comediSensor.device,
									   self.comediSensor.subdevice,
									   cmd,self.nchans,period)
		if ret: raise Exception("Error comedi_get_cmd_generic failed")
		#print "2"
		cmd.chanlist = mylist # adjust for our particular context
		cmd.chanlist_len = self.nchans
		cmd.scan_end_arg = self.nchans
		cmd.stop_arg=0
		cmd.stop_src=self.c.TRIG_NONE

		ret = self.c.comedi_command(self.comediSensor.device,cmd)
		#if ret !=0: raise Exception("comedi_command failed...")

	#Lines below are for initializing the format, depending on the comedi-card.
		data = os.read(self.fd,self.BUFSZ) # read buffer and returns binary data
		self.data_length=len(data)
		if self.maxdata[0]<=65536: # case for usb-dux-D
			n = self.data_length/2 # 2 bytes per 'H'
			self.format = `n`+'H'
		elif self.maxdata[0]>65536: #case for usb-dux-sigma
			n = self.data_length/4 # 2 bytes per 'H'
			self.format = `n`+'I'
			
	# init is over, start acquisition and stream
	def main(self):
		#print "3"
		try:
			while True:
				#print "4"
				array=np.zeros(self.nchans+1)
				data = os.read(self.fd,self.BUFSZ) # read buffer and returns binary
				if len(data)==self.data_length:
					datastr = struct.unpack(self.format,data)
					if len(datastr)==self.nchans: #if data not corrupted
						#print "5"
						array[0]=time.time()-self.t0
						for i in range(self.nchans):
							array[i+1]=self.c.comedi_to_phys((datastr[i]),
												self.range_ds[i],
												self.maxdata[i])
						if self.labels==None:
							self.Labels=[i for i in range(self.nchans+1)]
						#Array=pd.DataFrame([array],columns=self.labels)
						Array=OrderedDict(zip(self.labels,array))
						try:
							for output in self.outputs:
								output.send(Array)
						except TimeoutError:
							raise
						except AttributeError: #if no outputs
							pass

		except (Exception,KeyboardInterrupt) as e:	
			print "Exception in streamerComedi : ",
			self.comediSensor.close()
			#raise
