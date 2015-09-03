from ._meta import MasterBlock
from multiprocessing import Process, Pipe
import numpy as np
np.set_printoptions(threshold='nan', linewidth=500)
import time
import pandas as pd
import cv2
from ..technical import TechnicalCamera as tc


class VideoExtenso(MasterBlock): 
	"""
This class detects 4 spots, and evaluate the deformations Exx and Eyy.
	"""
	def __init__(self,camera="ximea",xoffset=0,yoffset=0,width=2048,height=2048,white_spot=True,display=True,labels=['t(s)','Exx ()', 'Eyy()']):
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
		go=False
		###################################################################### camera INIT with ZOI selection
		self.white_spot=white_spot
		self.labels=labels
		self.display=display
		self.border=4
		while go==False:
		# the following is to initialise the spot detection
			self.camera=tc(camera, {'enabled':True, 'white_spot':white_spot, 'border':self.border,'xoffset':xoffset,'yoffset':yoffset,'width':width,'height':height})
			self.minx=self.camera.minx
			self.maxx=self.camera.maxx
			self.miny=self.camera.miny
			self.maxy=self.camera.maxy
			self.NumOfReg=self.camera.NumOfReg
			self.L0x = self.camera.L0x
			self.L0y = self.camera.L0y
			self.thresh=self.camera.thresh
			self.Points_coordinates=self.camera.Points_coordinates
			self.width=self.camera.width
			self.height=self.camera.height
			self.xoffset=self.camera.xoffset
			self.yoffset=self.camera.yoffset
			self.exposure=self.camera.exposure
			self.gain=self.camera.gain
			if self.NumOfReg==4: 
				go=True
			else:	#	If detection goes wrong, start again
				print " Spots detected : ", self.NumOfReg		


	def barycenter_opencv(self,recv_):
		"""
		computation of the barycenter (moment 1 of image) on ZOI using OpenCV
		white_spot must be True if spots are white on a dark material
		"""
		# The median filter helps a lot for real life images ...
		while True:
			image,minx,miny=recv_.recv()[:]
			#print "minx ", minx
			bw=cv2.medianBlur(image,5)>self.thresh
			if not (self.white_spot):
				bw=1-bw
			M = cv2.moments(bw*255.)
			Px=M['m01']/M['m00']
			Py=M['m10']/M['m00'] 
			# we add minx and miny to go back to global coordinate:
			Px+=minx
			Py+=miny
			miny_, minx_, h, w= cv2.boundingRect((bw*255).astype(np.uint8)) # cv2 returns x,y,w,h but x and y are inverted
			maxy_=miny_+h
			maxx_=miny_+w
			# Determination of the new bounding box using global coordinates and the margin
			minx=minx-self.border+minx_
			miny=miny-self.border+miny_
			maxx=minx+self.border+maxx_
			maxy=miny+self.border+maxy_
			recv_.send([Px,Py,minx,miny,maxx,maxy])

	def main(self):
		"""
		main function, command the videoextenso and the motors
		"""
		self.camera.sensor.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain)
		j=0
		#t2=0
		#t3=0
		#t4=0
		#t5=0
		last_ttimer=time.time()
		first_display=True
		first=[True,True,True,True]
		proc_bary={}
		recv_={}
		send_={}
		#self.t0 = time.time()
		while True:
			try:	
				#t1=time.time()
				image = self.camera.sensor.getImage() # read a frame
				#t2_=time.time()
				#t2+=t2_-t1
				#print "image shape : ",np.shape(image)
				for i in range(0,self.NumOfReg): # for each spot, calulate the news coordinates of the center, based on previous coordinate and border.
					if first[i]:
						recv_[i],send_[i]=Pipe()
						proc_bary[i]=Process(target=self.barycenter_opencv,args=(recv_[i],))
						proc_bary[i].start()
						first[i]=False
						#print "i : ",i
						#print np.shape(image[self.minx[i]:self.maxx[i],self.miny[i]:self.maxy[i]]),self.minx[i],self.miny[i]
					send_[i].send([image[self.minx[i]:self.maxx[i],self.miny[i]:self.maxy[i]],self.minx[i],self.miny[i]])
				for i in range(0,self.NumOfReg):
					self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=send_[i].recv()[:]
					#self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=self.barycenter_opencv(image[self.minx[i]:self.maxx[i],self.miny[i]:self.maxy[i]],self.minx[i],self.miny[i])
				
				minx_=self.minx.min()
				miny_=self.miny.min()
				maxx_=self.maxx.max()
				maxy_=self.maxy.max()				
				Lx=100.*((self.Points_coordinates[:,0].max()-self.Points_coordinates[:,0].min())/self.L0x-1.)
				Ly=100.*((self.Points_coordinates[:,1].max()-self.Points_coordinates[:,1].min())/self.L0y-1.)
				self.Points_coordinates[:,1]-=miny_
				self.Points_coordinates[:,0]-=minx_
				Array=pd.DataFrame([[time.time()-self.t0,Lx,Ly]],columns=self.labels)
				#t3_=time.time()
				#t3+=t3_-t2_
				try:
					for output in self.outputs:
						output.send(Array)
				except AttributeError:
					pass
				#t4_=time.time()
				#t4+=t4_-t3_		
				if self.display:
					if first_display:
						self.plot_pipe_recv,self.plot_pipe_send=Pipe()
						proc=Process(target=self.plotter,args=())
						proc.start()
						first_display=False
					if j%90==0 and j>0: # every 80 round, send an image to the plot function below, that display the cropped image, LX, Ly and the position of the area around the spots
						self.plot_pipe_send.send([self.NumOfReg,self.minx-minx_,self.maxx-minx_,self.miny-miny_,self.maxy-miny_,self.Points_coordinates,self.L0x,self.L0y,image[minx_:maxx_,miny_:maxy_]])
				#t5_=time.time()
				#t5+=t5_-t4_
				if j%90==0 and j>0:
					t_now=time.time()
					print "FPS: ", 90/(t_now-last_ttimer)
					#print "bary :",t2/90.
					#print "eval coord :",t3/90.
					#print "send outputs :",t4/90.
					#print "send display :",t5/90.
					t2=0
					t3=0
					t4=0
					t5=0
					last_ttimer=t_now
				
				j+=1
			except KeyboardInterrupt:
				for i in range(0,self.NumOfReg):
					proc_bary[i].terminate()
				raise

	def plotter(self):
		data=self.plot_pipe_recv.recv() # receiving data
		NumOfReg=data[0]
		minx=data[1]
		maxx=data[2]
		miny=data[3]
		maxy=data[4]
		Points_coordinates=data[5]
		L0x=data[6]
		L0y=data[7]
		frame=data[-1]
		if self.white_spot:
			color=255
		else:
			color=0
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		for i in range(0,NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
			frame = cv2.rectangle(frame,(miny[i],minx[i]),(maxy[i]-1,maxx[i]-1),(color,0,0),1)
			frame = cv2.circle(frame,(int(Points_coordinates[i,1]),int(Points_coordinates[i,0])),1,(255-color,0,0),-1)
		cv2.imshow('frame',frame)
		cv2.waitKey(1)
		while True: # for every round, receive data, correct the positions of the rectangles/centers and the values of Lx/Ly , and refresh the plot.
			data=self.plot_pipe_recv.recv()
			NumOfReg=data[0]
			minx=data[1]
			maxx=data[2]
			miny=data[3]
			maxy=data[4]
			Points_coordinates=data[5]
			frame=data[-1]
			for i in range(0,NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
				frame = cv2.rectangle(frame,(miny[i],minx[i]),(maxy[i]-1,maxx[i]-1),(color,0,0),1)
				frame = cv2.circle(frame,(int(Points_coordinates[i,1]),int(Points_coordinates[i,0])),1,(255-color,0,0),-1)
			cv2.imshow('frame',frame)
			cv2.waitKey(1)