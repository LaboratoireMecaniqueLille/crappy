# coding: utf-8
#import numpy as np
import time
import comedi as c
#from multiprocessing import Array
#import os
#import sys, string, struct



class ComediSensor(object):
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
		
	 
	def new(self):
		""" Gather range and maxdata for all specified channels. This is only 
		called on init."""
		self.maxdata=[0]*self.nchans
		self.range_ds=[0]*self.nchans		
		for i in range(self.nchans):
			self.maxdata[i]=c.comedi_get_maxdata(self.device,self.subdevice,
										self.channels[i])
			self.range_ds[i]=c.comedi_get_range(self.device,self.subdevice,
									   self.channels[i],self.range_num[i])

	def getData(self,channel_number):
		"""Read the signal for desired channel"""
		data = c.comedi_data_read(self.device,self.subdevice,
							self.channels[channel_number],
							self.range_num[channel_number], c.AREF_GROUND)
		self.position=(c.comedi_to_phys(data[1],self.range_ds[channel_number],
					self.maxdata[channel_number])*self.gain[channel_number]+self.offset[channel_number])
		t=time.time()
		return (t, self.position)
			
	def close(self):
		"""Close the device."""
		c.comedi_cancel(self.device,self.subdevice)
		ret = c.comedi_close(self.device)
		if ret !=0: raise Exception('comedi_close failed...')
		
	#def _stream(self):
		#'''
		#[Deprecated]
		#Read the channels defined in chans, on the device/subdevice, 
		#and streams the values in the shared_array.
		#'''
		#fd = c.comedi_fileno(self.device)	# get a file-descriptor

		#self.BUFSZ = 10000	# buffer size
		#self.freq=8000	# acquisition frequency
	
		#nchans = len(self.channels)	# number of channels
		#self.shared_array= Array('f',np.arange(nchans))
		#self.aref =[c.AREF_GROUND]*nchans

		#mylist = c.chanlist(nchans)	# create a chanlist of length nchans
		#maxdata=[0]*(nchans)
		#range_ds=[0]*(nchans)

		#for index in range(nchans):	# pack informations into the chanlist
			#mylist[index]=c.cr_pack(self.channels[index],
						   #self.range_num[index], self.aref[index])
			#maxdata[index]=c.comedi_get_maxdata(self.device,
									   #self.subdevice,self.channels[index])
			#range_ds[index]=c.comedi_get_range(self.device,
									  #self.subdevice,self.channels[index],
									  #self.range_num[index])

		#cmd = c.comedi_cmd_struct()

		#period = int(1.0e9/self.freq)	# in nanoseconds
		#ret = c.comedi_get_cmd_generic_timed(self.device,self.subdevice,
									   #cmd,nchans,period)
		#if ret: raise Exception("Error comedi_get_cmd_generic failed")
			
		#cmd.chanlist = mylist # adjust for our particular context
		#cmd.chanlist_len = nchans
		#cmd.scan_end_arg = nchans
		#cmd.stop_arg=0
		#cmd.stop_src=c.TRIG_NONE

		#ret = c.comedi_command(self.device,cmd)
		#if ret !=0: raise Exception("comedi_command failed...")

	##Lines below are for initializing the format, depending on the comedi-card.
		#data = os.read(fd,self.BUFSZ) # read buffer and returns binary data
		#data_length=len(data)
		#if maxdata[0]<=65536: # case for usb-dux-D
			#n = data_length/2 # 2 bytes per 'H'
			#format = `n`+'H'
		#elif maxdata[0]>65536: #case for usb-dux-sigma
			#n = data_length/4 # 2 bytes per 'H'
			#format = `n`+'I'
			
	## init is over, start acquisition and stream
		##last_t=time.time()
		#try:
			#while True:
				#data = os.read(fd,self.BUFSZ) # read buffer and returns binary
				#if len(data)==data_length:
					#datastr = struct.unpack(format,data)
					#if len(datastr)==nchans: #if data not corrupted
						#for i in range(nchans):
							#self.shared_array[i]=c.comedi_to_phys((datastr[i]),
												#range_ds[i],maxdata[i])

		#except Exception as e:	
			#print "error in comediSensor : ", e
			#self.close()
			#raise