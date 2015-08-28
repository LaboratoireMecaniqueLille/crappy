from _meta import MasterBlock
import os
import time

class StreamerCamera(MasterBlock):
	"""
Children class of MasterBlock. Send frames through a Link object.
	"""
	def __init__(self,camera,freq=None,save=False,
			  save_directory="./images/"):
		"""
StreamerCamera(cameraSensor,freq=None,save=False,save_directory="./images/")

This block fetch images from a camera object, save and/or transmit them to 
another block. It can be triggered by a Link sending boolean or internally 
by defining the frequency.

Parameters:
-----------
camera : string, {"Ximea","Jai"}
	See sensor.cameraSensor documentation.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition device capability.
save : boolean
	Set to True if you want the block to save images.
save_directory : directory
	directory to the saving folder. If inexistant, will be created.
		"""
		print "streamer camera!!"
		import SimpleITK as sitk
		self.sitk = sitk
		if camera=="Ximea":
			from crappy.technical import Ximea
			self.CameraClass=Ximea
		elif camera=="Jai":
			from crappy.technical import Jai
			self.CameraClass=Jai
		self.freq=freq
		self.save=save
		self.i=0
		self.save_directory=save_directory
		if not os.path.exists(self.save_directory) and self.save:
			os.makedirs(self.save_directory)

	def main(self):
		self.cameraSensor=self.CameraClass()
		try:
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
							pass
					timer=time.time()
					img=self.cameraSensor.sensor.getImage()
					#print "internal"
				if trigger=="external":
					if self.inputs[0].recv(): # wait for a signal
						img=self.cameraSensor.sensor.getImage()
					#print "external"
				if self.save:
					image=self.sitk.GetImageFromArray(img)
					self.sitk.WriteImage(image,
						  self.save_directory+"img_%.6d.tiff" %(self.i))
					self.i+=1
				try:
					for output in self.outputs:
						output.send(img)
						#print "sending :", img[0][0]
				except AttributeError:
					pass

		except (KeyboardInterrupt):	
			self.cameraSensor.sensor.close()
