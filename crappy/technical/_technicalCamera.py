# coding: utf-8
#from ._meta import cameraSensor
import time
from multiprocessing import Process,Pipe
from . import getCameraConfig

class TechnicalCamera(object):
	"""
Opens a camera device and initialise it.
	"""
	def __init__(self, camera="ximea",numdevice=0, videoextenso={}):
		"""
This Class opens a device and runs the initialisation sequence (CameraInit). 
It then closes the device and keep the parameters in memory for later use.

Parameters
----------
cam : {'ximea','jai'}, default = 'ximea'
	Name of the desired camera device.
numdevice : int, default = 0
	Number of the desired device.
videoextenso : dict
	dict of parameters that you can use to pass informations.

		* 'enabled' : Bool
			Set True if you need the videoextenso.
		* 'white_spot' : Bool
			Set to True if your spots are white on a dark material.
		* 'border' : int, default = 4
			Size of the border for spot detection
		* 'xoffset' : int
			Offset for the x-axis.
		* 'yoffset' : int
			Offset for the y-axis
		* 'height' : int
			Height of the image, in pixels.
		* 'width : int
			Width of the image, in pixels.
		"""
		try:
			module = __import__("crappy.sensor", fromlist=[camera.capitalize()])
			CameraClass= getattr(module, camera.capitalize())
		except Exception as e:
			print "{0}".format(e), " : Unreconized camera\n"
			import sys
			sys.exit()
			raise
		try:
			module = __import__("crappy.sensor.clserial", fromlist=[camera.capitalize()+"Serial"])
			codeClass = getattr(module, camera.capitalize()+"Serial")
			from crappy.sensor.clserial import ClSerial as cl
			ser = codeClass()
			self.serial = cl(ser)
		except Exception as e:
			print "{0}".format(e)
			self.serial = None
		#print "module, cameraclass, serial : ", module, CameraClass, self.serial
        #initialisation:
		self.sensor = CameraClass(numdevice=numdevice, serial= self.serial)
		self.videoextenso = videoextenso
		recv_pipe,send_pipe=Pipe()
		print "lauching camera config..."
		proc_test=Process(target=getCameraConfig,args=(self.sensor,self.videoextenso,send_pipe))
		proc_test.start()
		data=recv_pipe.recv()
		print "data received, config done."
		if self.videoextenso['enabled']:
			self.exposure,self.gain,self.width,self.height,self.xoffset,self.yoffset, self.minx, self.maxx, self.miny, self.maxy, self.NumOfReg, self.L0x, self.L0y,self.thresh,self.Points_coordinates =data[:]
		else:
			self.exposure,self.gain,self.width,self.height,self.xoffset,self.yoffset=data[:]
			
		### here we should modify height, width and others in cameraclass().sensor #WIP
		
		
		#self.sensor.exposure=self.exposure
		#self.sensor.gain=self.gain
		proc_test.terminate()
		#self.cam.new(exposure=exposure, gain=gain, yoffset=yoffset, xoffset=xoffset, height=height, width=width)
	
	#def _interface(self, send_pipe, camera):
		#settings = getCameraConfig(camera, self.videoextenso)
		#send_pipe.send(settings)
		
	def __str__(self):
		return self.cam.__str__()