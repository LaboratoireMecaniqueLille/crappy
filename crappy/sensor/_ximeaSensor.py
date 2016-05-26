# coding: utf-8
from ._meta import cameraSensor
import numpy as np
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from os import path
here = path.abspath(path.dirname(__file__))
import ximeaModule as xi
# =======
# #import numpy as np
# #import ctypes
# #import numpy as np
# #from numpy.ctypeslib import ndpointer
# #from os import path
# #here = path.abspath(path.dirname(__file__))
# #import ximeaModule as xi


# #try :
# import cv2 as xi
# #except ImportError: 
# 	#print "WARNING : OpenCV2 is not installed, some functionalities may crash"
# >>>>>>> master
import time
import platform
from sys import stdout
    
def resettable(f, *args, **kwargs):
    """
    Decorator for resetting the camera device. Not working yet.
    """
    import copy
    def __init_and_copy__(self, *args, **kwargs):
        if(not platform.system()=="Linux"):
            return f(self, **kwargs)
        f(self, **kwargs)
        self.__original_dict__ = copy.deepcopy(self.__dict__)
        def reset(o = self):
            o.__dict__ = o.__original_dict__

        self.reset = reset
    return __init_and_copy__
    
class Ximea(cameraSensor.CameraSensor):
	"""
	Camera class for ximea devices, this class should inherit from CameraObject
	
	Contains all the methods to open a device, resize the Zone Of Interest, and
	grab frames.
	
	Parameters
	----------
	numdevice : int, default = 0
		Number of your device.
	framespersec : int or float or None, default = None
		The wanted frequency for grabbing frame. DOESN'T WORK at the moment.
	external_trigger : bool, default = False
		Define to True if you want to trigg the acquyisition of a frame externally.
	data_format : int, default = 0
		Value must be in [0:7]. See documentation for more informations.
	"""
	@resettable
	def __init__(self, numdevice=0, framespersec=None, external_trigger=False, data_format=0):
		self.quit=False
		self.FPS=framespersec
		#self.framespersec=self.FPS
		self.numdevice = numdevice
		self.external_trigger=external_trigger
		self.data_format=data_format
		#self.actuator=None
		self._defaultWidth = 2048
		self._defaultHeight = 2048
		self._defaultXoffset = 0
		self._defaultYoffset = 0
		self._defaultExposure = 10000
		self._defaultGain= 0
		self.nbi = 0

	def new(self, exposure=10000, width=2048, height=2048, xoffset=0, yoffset=0, gain=0):
		"""
		This method opens the ximea device and return a camera object.
		"""
		#Ximea devices start at 1100. 1100 => device 0, 1101 => device 1
		#self.sensor=_ximeaSensor.XimeaSensor(self.numdevice, self.exposure, self.gain, self.width, self.height, self.xoffset, self.yoffset, self.framespersec, self.external_trigger, self.data_format)
		if(platform.system()=="Linux"):
			nd, fps, et, df = self.numdevice,self.FPS,self.external_trigger,self.data_format
			self.reset()
			self.__init__(numdevice=nd, framespersec = fps, external_trigger=et, data_format=df)
		
		GLOBAL_ENABLE_FLAG = True
		
		#self.ximea = xi.VideoCapture(self.numdevice) # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1 
		self.ximea = xi.VideoCapture(self.numdevice) # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1 
		if self.external_trigger==True:	# this condition activate the trigger mode
                    self.ximea.addTrigger(1000000, True)
		self.ximea.set(xi.CAP_PROP_XI_DATA_FORMAT,self.data_format) #0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW

		if self.data_format ==1 or self.data_format==6: #increase the FPS in 10 bits
			self.ximea.set(xi.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH,10)
			self.ximea.set(xi.CAP_PROP_XI_DATA_PACKING,1)
		
		self.ximea.set(xi.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
		#self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.width);	# doesn't work for this one
		#self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height); # reducing this one allows one to increase the FPS
		
		#self.cam.set(cv2.CAP_PROP_EXPOSURE,self.exposure) # setting up exposure
		#self.cam.set(cv2.CAP_PROP_GAIN,self.gain) #setting up gain
		#ret, frame= self.cam.read()
		self.width = width
		self.height = height
		self.xoffset = xoffset
		self.yoffset = yoffset
		self.exposure = exposure
		self.gain=gain

	def getImage(self):
		"""
		This method get a frame on the selected camera and return a ndarray 
		If the camera breaks down, it reinitializes it, and tries again.
		"""
		self.nbi = self.nbi+1
		try:
			ret, frame = self.ximea.read()
		except KeyboardInterrupt:
			print "KeyboardInterrupt, closing camera ..."
			self.close()
			self.quit=True

		try:
			#stdout.write("\rret = {}".format(ret))
			#stdout.flush()
			if ret==True:
                                data = frame.get('data')
				return data

			# if ret==1:
			# 	#print "sending frame"
			# 	#return frame.get('data')
			# 	return frame

			elif not(self.quit):
				print "restarting camera..."
				time.sleep(2)
				#expo, wi, he, xoff,yoff,ga=self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain
				#self.reset()
				#self.__init__()
				#self.new(expo, wi, he, xoff,yoff,ga) # Reset the camera instance
				self.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain) # Reset the camera instance
				return self.getImage()
		except UnboundLocalError: # if ret doesn't exist, because of KeyboardInterrupt
			print "ximea quitting, probably because of KeyBoardInterrupt"
			pass
		
	def close(self):
		"""
		This method close properly the frame grabber.
		It releases the allocated memory and stops the acquisition.
		"""
		print "closing camera..."
		if self.ximea.isOpened():
			self.ximea.release()
			print "cam closed"
		else:
			print "cam already closed"

		
	def reset_ZOI(self):
		"""Re-initialize the Zone Of Interest"""
		self.yoffset = self._defaultYoffset
		self.xoffset = self._defaultXoffset
		self.height = self._defaultHeight
		self.width = self._defaultWidth
	
	def set_ZOI(self, width, height, xoffset, yoffset):
		"""Define the Zone Of Interest"""
		self.yoffset = yoffset
		self.xoffset = xoffset
		self.width = width
		self.height = height
		
	@property
	def height(self):
		"""Property. Set / get the current height."""
		return self._height
        
	@height.setter
	def height(self,height):
		#print "height setter : ", height
		self._height=((int(height)/2)*2)
		self.ximea.set(xi.CAP_PROP_FRAME_HEIGHT,self.height)

	@property
	def width(self):
		"""Property. Set / get the current width."""
		return self._width
    
	@width.setter
	def width(self,width):
		#print "width setter : ", width
		self._width=(int(width)-int(width)%4)
		self.ximea.set(xi.CAP_PROP_FRAME_WIDTH,self.width)

	@property
	def yoffset(self):
		"""Property. Set / get the current yoffset."""
		return self._yoffset

	@yoffset.setter
	def yoffset(self,yoffset):
		#print "yoffset setter : ", yoffset
		y_offset = ((int(yoffset)/2)*2)
		self._yoffset= y_offset
		self.ximea.set(xi.CAP_PROP_XI_OFFSET_Y,y_offset)
        
	@property
	def xoffset(self):
		"""Property. Set / get the current xoffset."""
		return self._xoffset
    
	@xoffset.setter
	def xoffset(self,xoffset):
		#print "xoffset setter : ", xoffset
		x_offset = (int(xoffset)-int(xoffset)%4)
		self._xoffset= x_offset 
		self.ximea.set(xi.CAP_PROP_XI_OFFSET_X, x_offset)
	
	@property
	def exposure(self):
		"""Property. Set / get the current exposure."""
		return self._exposure
		
	@exposure.setter
	def exposure(self, exposure):
		self.ximea.set(xi.CAP_PROP_EXPOSURE,exposure)
		self._exposure = exposure
		
	@property
	def gain(self):
		"""Property. Set / get the current gain."""
		return self._gain
		
	@gain.setter
	def gain(self, gain):
		self.ximea.set(xi.CAP_PROP_GAIN,gain)
		self._gain= gain
		
		
	def __str__(self):
		return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} \n X offset: {5} \n Y offset: {6}".format(self.exposure, self.FPS, self.numdevice, self.width, self.height, self.xoffset, self.yoffset)
	      
	@property
	def name(self):
	  return "ximea"
