#from ._meta import cameraSensor
import time
from multiprocessing import Process,Pipe
from . import getCameraConfig


class TechnicalCamera():
	"""
	Camera class for ximea devices, this class should inherit from CameraObject
	"""
	def __init__(self, camera="ximea", videoextenso={}):
		try:
			module = __import__("crappy.sensor", fromlist=[camera.capitalize()])
			CameraClass= getattr(module, camera.capitalize())
		except Exception as e:
			print "%s "%e, " Unreconized camera\n"
			sys.exit()
			raise
		
        #initialisation:
		self.sensor = CameraClass()
		self.videoextenso = videoextenso
		recv_pipe,send_pipe=Pipe()
		proc_test=Process(target=getCameraConfig,args=(self.sensor,self.videoextenso,send_pipe))
		proc_test.start()
		data=recv_pipe.recv()
		print "data received!"
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
		"""
		This method prints out the attributes values
		"""
		return self.cam.__str__()