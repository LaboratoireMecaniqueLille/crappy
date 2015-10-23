from _meta import MasterBlock
import os
import time
from crappy.technical import TechnicalCamera as tc

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
		#if camera=="Ximea":
			#from crappy.technical import Ximea
			#self.CameraClass=Ximea
		#elif camera=="Jai":
			#from crappy.technical import Jai
			#self.CameraClass=Jai
		self.camera=tc(camera, videoextenso={'enabled':False,'xoffset':0,'yoffset':0,'width':2048,'height':2048})
		self.freq=freq
		self.save=save
		self.i=0
		self.save_directory=save_directory
		if not os.path.exists(self.save_directory) and self.save:
			os.makedirs(self.save_directory)

	def main(self):
		print "streamer camera!!" , os.getpid()
		self.camera.sensor.new()
		try:
			_a=self.inputs[:]
			trigger="external"
		except AttributeError:
			trigger="internal"
		timer=time.time()
		try:
			#j=1
			#delta=0
			#t_max=0
			#t_mean=0
			#k=1
			while True:
				if trigger=="internal":
					if self.freq!=None:
						while time.time()-timer< 1./self.freq:
							pass
					timer=time.time()
					img=self.camera.sensor.getImage()
					#print "internal"
					if self.save:
						image=self.sitk.GetImageFromArray(img)
						self.sitk.WriteImage(image,
							self.save_directory+"img_%.6d.tiff" %(self.i))
						self.i+=1
				elif trigger=="external":
					#print " waiting for data"
					#t_1=time.time()
					Data = self.inputs[0].recv() # wait for a signal
					#t_recv=time.time()-t_1
					#t_max=max(t_max,t_recv)
					#t_mean+=t_recv
					#if k%10==0:
						#print "t_max, t_mean cam: ", t_max,t_mean/k
						#t_max=0
					#k+=1
					#print "data received"
					if Data is not None:
						img=self.camera.sensor.getImage()
						t=time.time()-self.t0
						#ret=self.agilentSensor.getData()
						#if ret == False:
							#ret=np.nan
						#Data['t_camera'] = Series((time.time()-self.t0), index=Data.index) # add one column
						#Data[self.labels[1]] = Series((ret), index=Data.index) # add one column
						#delta+=(time.time()-self.t0)-Data['t(s)'][0]
						#if j%100==0:
							#print "camera delta time : ", delta/j
						#j+=1
						if self.save:
							image=self.sitk.GetImageFromArray(img)
							self.sitk.WriteImage(image,
								self.save_directory+"img_%.6d_cycle%09.1f.tiff" %(self.i,Data['cycle'][0]))
							self.i+=1
				try:
					#print "top1"
					if trigger=="internal" or Data is not None:
						for output in self.outputs:
							output.send(img)
						#print "sending :", img[0][0]
				except AttributeError: # if no output or img not defined
					#print "top2"
					pass

		except (Exception,KeyboardInterrupt) as e:	
			print "Exception in streamerCamera : ",
			self.camera.sensor.close()
			raise
			
