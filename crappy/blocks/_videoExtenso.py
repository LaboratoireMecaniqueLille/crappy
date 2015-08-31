from ._meta import MasterBlock
from multiprocessing import Process, Pipe
import os
import numpy as np
np.set_printoptions(threshold='nan', linewidth=500)
import time
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import struct
import pandas as pd
import sys
from skimage.segmentation import clear_border
from skimage.morphology import label,erosion, square,dilation
from skimage.measure import regionprops
from skimage.filter import threshold_otsu, rank#, threshold_yen
import cv2
import SimpleITK as sitk
from ..technical import TechnicalCamera as tc


#def crop(image, x1, x2, y1, y2):
    #"""
    #Return the cropped image at the x1, x2, y1, y2 coordinates
    #"""
    #if x2 == -1:
        #x2=image.shape[1]-1
    #if y2 == -1:
        #y2=image.shape[0]-1

    #mask = np.zeros(image.shape)
    #mask[y1:y2+1, x1:x2+1]=1
    #m = mask>0

    #return image[m].reshape((y2+1-y1, x2+1-x1))


class VideoExtenso(MasterBlock): 
	"""
This class detects 4 spots, and evaluate the deformations Exx and Eyy.
	"""
	def __init__(self,camera="ximea",white_spot=True,display=True,labels=['t(s)','Exx ()', 'Eyy()']):
		"""
VideoExtenso(camera,white_spot=True,labels=['t(s)','Exx ()', 'Eyy()'],display=True)

Detects 4 spots, and evaluate the deformations Exx and Eyy. Can display the 
image with the center of the spots.

Parameters
----------
camera : string, {"Ximea","Jai"},default=Ximea
	See sensor.cameraSensor documentation.
white_spot : Boolean, default=True
	Set to False if you have dark spots on a light surface.
display : Boolean, default=True
	Set to False if you don't want to see the image with the spot detected.
labels : list of string, default = ['t(s)','Exx ()', 'Eyy()']

Returns:
--------
Panda Dataframe with time and deformations Exx and Eyy.
		"""
		self.cv2=cv2 # remove this
		self.sitk=sitk  #remove this 
		go=False
		###################################################################### camera INIT with ZOI selection
		self.white_spot=white_spot
		self.labels=labels
		self.display=display
		self.border=4
		#videoextenso = {'enabled'=True, 
		while go==False:
		# the following is to initialise the spot detection
			self.camera=tc(camera, {'enabled':True, 'white_spot':white_spot, 'border':self.border})
			self.minx=self.camera.minx
			self.maxx=self.camera.maxx
			self.miny=self.camera.miny
			self.maxy=self.camera.maxy
			self.NumOfReg=self.camera.NumOfReg
			self.L0x = self.camera.L0x
			self.L0y = self.camera.L0y
			self.thresh=self.camera.thresh
			self.Points_coordinates=self.camera.Points_coordinates
			if self.NumOfReg==4: 
				go=True
			else:	#	If detection goes wrong, start again
				print " Spots detected : ", self.NumOfReg
				#self.camera.sensor.close()

		#image=self.camera.sensor.getImage()
		#	data for re-opening the camera device
		#self.numdevice=self.camera.sensor.numdevice
		#self.exposure=self.camera.sensor.exposure
		#self.gain=self.camera.sensor.gain
		#self.width=self.camera.sensor.width
		#self.height=self.camera.height
		#self.xoffset=self.camera.xoffset
		#self.yoffset=self.camera.yoffset
		#self.external_trigger=self.camera.sensor.external_trigger
		#self.data_format=self.camera.sensor.data_format
		#self.camera.sensor.close()
		


	def barycenter_opencv(self,image,minx,miny):
		"""
		computation of the barycenter (moment 1 of image) on ZOI using OpenCV
		white_spot must be True if spots are white on a dark material
		"""
		# The median filter helps a lot for real life images ...
		bw=self.cv2.medianBlur(image,5)>self.thresh
		if not (self.white_spot):
			bw=1-bw
		M = self.cv2.moments(bw*255.)
		Px=M['m01']/M['m00']
		Py=M['m10']/M['m00'] 
		# we add minx and miny to go back to global coordinate:
		Px+=minx
		Py+=miny
		miny_, minx_, h, w= self.cv2.boundingRect((bw*255).astype(np.uint8)) # cv2 returns x,y,w,h but x and y are inverted
		maxy_=miny_+h
		maxx_=miny_+w
		# Determination of the new bounding box using global coordinates and the margin
		minx=minx-self.border+minx_
		miny=miny-self.border+miny_
		maxx=minx+self.border+maxx_
		maxy=miny+self.border+maxy_
		return Px,Py,minx,miny,maxx,maxy

	def main(self):
		"""
		main function, command the videoextenso and the motors
		"""
		self.camera.sensor.new()
		j=0
		last_ttimer=time.time()
		first_display=True
		self.t0 = time.time()
		while True:
			try:
				image = self.camera.sensor.getImage() # read a frame
				for i in range(0,self.NumOfReg): # for each spot, calulate the news coordinates of the center, based on previous coordinate and border.
					self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=self.barycenter_opencv(image[self.minx[i]:self.maxx[i],self.miny[i]:self.maxy[i]],self.minx[i],self.miny[i])
				minx_=self.minx.min()
				miny_=self.miny.min()
				maxx_=self.maxx.max()
				maxy_=self.maxy.max()
				Lx=100.*((self.Points_coordinates[:,0].max()-self.Points_coordinates[:,0].min())/self.L0x-1.)
				Ly=100.*((self.Points_coordinates[:,1].max()-self.Points_coordinates[:,1].min())/self.L0y-1.)
				self.Points_coordinates[:,1]-=miny_
				self.Points_coordinates[:,0]-=minx_
				Array=pd.DataFrame([[time.time()-self.t0,Lx,Ly]],columns=self.labels)
				try:
					for output in self.outputs:
						output.send(Array)
				except AttributeError:
					pass
						
				if self.display:
					if first_display:
						self.plot_pipe_recv,self.plot_pipe_send=Pipe()
						proc=Process(target=self.plotter,args=())
						proc.start()
						first_display=False
					if j%50==0 and j>0: # every 80 round, send an image to the plot function below, that display the cropped image, LX, Ly and the position of the area around the spots
						self.plot_pipe_send.send([self.NumOfReg,self.minx-minx_,self.maxx-minx_,self.miny-miny_,self.maxy-miny_,self.Points_coordinates,self.L0x,self.L0y,image[minx_:maxx_,miny_:maxy_]])
						t_now=time.time()
						print "FPS: ", 50/(t_now-last_ttimer)
						last_ttimer=t_now
				
				j+=1
			except KeyboardInterrupt:
				raise

	def plotter(self):
		#import cv2
		#self.cv2=reload(self.cv2)
		#print "I'm here!!"
		#rec={}
		#center={}
		#plt.ion()
		#print "top1"
		#time.sleep(2)
		#print "top11"
		#fig=plt.figure(2)
		#fig=cv2.figure(2)
		#print "top12"
		#ax=fig.add_subplot(111)
		#print "top2"
		data=self.plot_pipe_recv.recv() # receiving data
		#print "data received"
		NumOfReg=data[0]
		#print "1"
		minx=data[1]
		maxx=data[2]
		miny=data[3]
		#print "2"
		maxy=data[4]
		Points_coordinates=data[5]
		L0x=data[6]
		L0y=data[7]
		frame=data[-1]
		if self.white_spot:
			color=255
		else:
			color=0
		#height, width = frame.shape
		#frame = self.cv2.resize(frame,(2*width, 2*height), interpolation = self.cv2.INTER_CUBIC)
		#print "data processed"
		#im = plt.imshow(frame,cmap='gray')
		#self.cv2.destroyAllWindows()
		#print "1"
		#self.cv2.waitKey(1)
		#print "2"
		
		self.cv2.namedWindow('frame',self.cv2.WINDOW_NORMAL)
		#print "window!"
		
		for i in range(0,NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
			frame = self.cv2.rectangle(frame,(miny[i],minx[i]),(maxy[i]-1,maxx[i]-1),(color,0,0),1)
			frame = self.cv2.circle(frame,(int(Points_coordinates[i,1]),int(Points_coordinates[i,0])),1,(255-color,0,0),-1)
			#rect = mpatches.Rectangle((miny[i], frame.shape[0]-maxx[i]), maxy[i] - miny[i], maxx[i] - minx[i],fill=False, edgecolor='red', linewidth=1)
			#rec[i]=ax.add_patch(rect)
			#center[i],= ax.plot(Points_coordinates[i,1],frame.shape[0]-Points_coordinates[i,0],'+g',markersize=5) # coordinate here are not working, needs to be fixed
			
		#for i in range(0,NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
			#rect = mpatches.Rectangle((miny[i], frame.shape[0]-maxx[i]), maxy[i] - miny[i], maxx[i] - minx[i],fill=False, edgecolor='red', linewidth=1)
			#rec[i]=ax.add_patch(rect)
			#center[i],= ax.plot(Points_coordinates[i,1],frame.shape[0]-Points_coordinates[i,0],'+g',markersize=5) # coordinate here are not working, needs to be fixed
		#im.set_extent((0,frame.shape[1],0,frame.shape[0])) # adjust the width and height of the plotted figure depending of the size of the received image
		#ax.set_xlabel("This is the Y Axis")
		#ax.set_ylabel("This is the X Axis")
		#Exx = "Exx = 0 %%"
		#Eyy = "Eyy = 0 %%"
		#exx=ax.text(1, 1, Exx, fontsize=12,color='white', va='bottom') # plot some text with the Lx and Ly values on the images
		#eyy=ax.text(1, 11, Eyy, fontsize=12,color='white', va='bottom')
		#fig.canvas.draw()
		#plt.show(block=False)
		#print "rect"
		self.cv2.imshow('frame',frame)
		self.cv2.waitKey(1)
		while True: # for every round, receive data, correct the positions of the rectangles/centers and the values of Lx/Ly , and refresh the plot.
			data=self.plot_pipe_recv.recv()
			#print "data received"
			NumOfReg=data[0]
			minx=data[1]
			maxx=data[2]
			miny=data[3]
			maxy=data[4]
			Points_coordinates=data[5]
			frame=data[-1]
			#height, width = frame.shape
			#frame = self.cv2.resize(frame,(2*width, 2*height), interpolation = self.cv2.INTER_LINEAR)
			#j+=1
			#for i in range(0,NumOfReg): 
				#rec[i].set_bounds(miny[i],frame.shape[0]-maxx[i],maxy[i] - miny[i],maxx[i] - minx[i])
				#center[i].set_xdata(Points_coordinates[i,1])
				#center[i].set_ydata(frame.shape[0]-Points_coordinates[i,0])
			#Lx=Points_coordinates[:,0].max()-Points_coordinates[:,0].min()
			#Ly=Points_coordinates[:,1].max()-Points_coordinates[:,1].min()
			#Exx = "Exx = %2.2f %%"%(100.*(Lx/L0x-1.))
			#Eyy = "Eyy = %2.2f %%"%(100.*(Ly/L0y-1.))
			#exx.set_text("Exx = %2.2f %%"%(100.*(Lx/L0x-1.)))
			#eyy.set_text("Eyy = %2.2f %%"%(100.*(Ly/L0y-1.)))
			#im.set_array(frame)
			#im.set_extent((0,frame.shape[1],0,frame.shape[0]))
			#fig.canvas.draw()
			#font = self.cv2.FONT_HERSHEY_SIMPLEX
			#self.cv2.putText(frame,"Exx",(1,frame.shape[0]-2), font, 0.2,(255,255,255),1,self.cv2.LINE_AA)
			for i in range(0,NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
				frame = self.cv2.rectangle(frame,(miny[i],minx[i]),(maxy[i]-1,maxx[i]-1),(color,0,0),1)
				frame = self.cv2.circle(frame,(int(Points_coordinates[i,1]),int(Points_coordinates[i,0])),1,(255-color,0,0),-1)
			self.cv2.imshow('frame',frame)
			self.cv2.waitKey(1)
			#plt.show(block=False)