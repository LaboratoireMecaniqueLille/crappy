from ._meta import cameraSensor
import numpy as np
import cv2
import time

class XimeaSensor(cameraSensor.CameraSensor):
	"""
	Camera class for ximea devices, this class should inherit from CameraObject
	"""
	def __init__(self, numdevice=0, exposure=10000, gain=0, width=2048, height=2048, xoffset=0, yoffset=0, framespersec=None, external_trigger=False, data_format=0):
		self.exposure= exposure
		self.gain=gain
		self.FPS=framespersec
		self.framespersec=self.FPS
		self.numdevice = numdevice
		self.width=width
		self.height=height
		self.xoffset=xoffset
		self.yoffset=yoffset
		self.cam = None
		self.external_trigger=external_trigger
		self.data_format=data_format
		self.new()
		
	def new(self):
		"""
		this method opens the ximea device. Ximea devices start at 1100. 1100 => device 0, 1101 => device 1
		And return a camera object
		"""
		#try:
		self.cam = cv2.VideoCapture(cv2.CAP_XIAPI + self.numdevice) # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1 

		if self.external_trigger==True:	# this condition activate the trigger mode
			self.cam.set(cv2.CAP_PROP_XI_TRG_SOURCE,1)
			self.cam.set(cv2.CAP_PROP_XI_GPI_SELECTOR,1)
			self.cam.set(cv2.CAP_PROP_XI_GPI_MODE,1)
			
		self.cam.set(cv2.CAP_PROP_XI_DATA_FORMAT,self.data_format) #0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW	

		if self.data_format ==1 or self.data_format==6: #increase the FPS in 10 bits
			self.cam.set(cv2.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH,10)
			self.cam.set(cv2.CAP_PROP_XI_DATA_PACKING,1)
		
		self.cam.set(cv2.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
		self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.width);	# doesn't work for this one
		self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height); # reducing this one allows one to increase the FPS
		
		self.cam.set(cv2.CAP_PROP_EXPOSURE,self.exposure) # setting up exposure
		self.cam.set(cv2.CAP_PROP_GAIN,self.gain) #setting up gain
		ret, frame= self.cam.read()
		#except Exception as e:
			#print e
			#self.close()
			#self.new()

	
	def getImage(self):
		"""
		This method get a frame on the selected camera and return a ndarray 
		If the camera breaks down, it reinitializes it, and tries again.
		"""
		try:
			#print "getting image ...."
			#print self.cam
			ret, frame = self.cam.read()
			if ret:
				#print "frame : ",frame[0][0]
				return frame
			else:
				raise Exception('failed to grab a frame\n')

		except Exception as e:
			print e
			self.close()
			self.new() # Reset the camera instance
			return self.getImage()

			
	def setExposure(self, exposure):
		"""
		this method changes the exposure of the camera
		and set the exposure attribute
		"""
		self.cam.set(cv2.CAP_PROP_EXPOSURE,exposure)
		self.exposure = exposure
	
	def close(self):
		"""
		This method close properly the frame grabber
		It releases the allocated memory and stops the acquisition
		"""
		print "closing camera..."
		self.cam.release()
		print "cam closed"

	def __str__(self):
		"""
		This method prints out the attributes values
		"""
		return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} \n X offset: {5} \n Y offset: {6}".format(self.exposure, self.FPS, self.numdevice, self.width, self.height, self.xoffset, self.yoffset)
