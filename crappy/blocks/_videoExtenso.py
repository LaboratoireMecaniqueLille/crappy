# coding: utf-8
from ._meta import MasterBlock
from multiprocessing import Process, Pipe
import numpy as np
np.set_printoptions(threshold='nan', linewidth=500)
import time
#import pandas as pd
import cv2
from ..links._link import TimeoutError
from ..technical import TechnicalCamera as tc
#import SimpleITK as sitk # only for testing
from skimage.filter import threshold_otsu, rank
from skimage.measure import regionprops
from skimage.morphology import label,erosion, square,dilation
from skimage.segmentation import clear_border
from collections import OrderedDict
from sys import stdout
try:
	import pyglet
	import glob
	import random
except ImportError:
	print "no sound module installed"
	

class VideoExtenso(MasterBlock): 
	"""
Detects spots (1,2 or 4) on images, and evaluate the deformations Exx and Eyy.
	"""
	def __init__(self,camera="ximea",numdevice=0,xoffset=0,yoffset=0,width=2048,height=2048,white_spot=True,display=True,update_tresh=False,labels=['t(s)','Lx','Ly','Exx(%)','Eyy(%)']):
		"""
Detects 1/2/4 spots, and evaluate the deformations Exx and Eyy. Can display the 
image with the center of the spots.

4 spots mode : deformations are evaluated on the distance between centers of spots.
2 spots mode : same, but deformation is only reliable on 1 axis.
1 spot : deformation is evaluated on the major/minor axis of a theorical ellipse 
around the spot, projected over axis x and y. Results are less precise if your
spot isn't big enough, but it is easier on smaller sample to only have 1 spot.

Note that if this block lose the spots, it will play a song in the '/home/' 
repository. You need a .wav sound, python-pyglet and python-glob. This can be 
usefull if you have a long test to do, as the script doesn't stop when losing 
spots. Not to mention it is fun.

Parameters
----------
camera : string, {"Ximea","Jai"},default=Ximea
	See sensor.cameraSensor documentation.
numdevice : int, default=0
	If you have multiple camera plugged, select the correct one.
xoffset: int, default =0
	Offset on the x axis.
yoffset: int, default =0
	Offset on the y axis.
width: int, default = 2048
	Width of the image.
height: int, default = 2048
	Height of the image.
white_spot : Boolean, default=True
	Set to False if you have dark spots on a light surface.
display : Boolean, default=True
	Set to False if you don't want to see the image with the spot detected.
update_tresh : Boolean, default=False
	Set to True if you want to re-evaluate the threshold for every new image.
	Updside is that it allows you to follow more easily your spots even if your 
	light changes. Downside is that it will change the area and possibly the 
	shape of the spots, wich may inscrease the noise on the deformation and 
	artificially change its value. This is especially true with a single spot 
	configuration.
labels : list of string, default = ['t(s)','Lx','Ly','Exx(%)','Eyy(%)']
	Labels of your output. Order is important.

Returns
-------
dict : OrderedDict


	time : float
		Time of the measure, relative to t0.
	Lx : float
		Lenght (in pixels) of the spot.
	Ly : float
		Width (in pixels) of the spot.
	Exx : float
		Deformation = Lx/L0x
	Eyy : float
		Deformation = Lxy/L0y
		"""
		go=False
		###################################################################### camera INIT with ZOI selection
		self.white_spot=white_spot
		self.labels=labels
		self.display=display
		self.border=4
		self.numdevice=numdevice
		self.update_tresh=update_tresh
		while go==False:
		# the following is to initialise the spot detection
			self.camera=tc(camera,self.numdevice,{'enabled':True, 'white_spot':white_spot, 'border':self.border,'xoffset':xoffset,'yoffset':yoffset,'width':width,'height':height})
			self.minx=self.camera.minx
			self.maxx=self.camera.maxx
			self.miny=self.camera.miny
			self.maxy=self.camera.maxy
			self.NumOfReg=self.camera.NumOfReg
			self.L0x = self.camera.L0x
			self.L0y = self.camera.L0y
			self.thresh=self.camera.thresh
			#print "tresh initial :", self.thresh
			self.Points_coordinates=self.camera.Points_coordinates
			self.width=self.camera.width
			self.height=self.camera.height
			self.xoffset=self.camera.xoffset
			self.yoffset=self.camera.yoffset
			self.exposure=self.camera.exposure
			self.gain=self.camera.gain
			if self.NumOfReg==4 or self.NumOfReg==2 or self.NumOfReg==1: 
				go=True
			else:	#	If detection goes wrong, start again, may not be usefull now ?
				print " Spots detected : ", self.NumOfReg	
		print "initialisation done, starting acquisition"
			# following is for tests only
			#self.save_directory="/home/biaxe/Bureau/Publi/"
			#fo=open(self.save_directory+"L0.txt","a")		# "a" for appending
			#fo.seek(0,2)		#place the "cursor" at the end of the file
			#data_to_save="L0x : "+str(self.L0x)+"\n"+"Loy : "+str(self.L0y)
			#fo.write(data_to_save)
			#fo.close()

	def barycenter_opencv(self,recv_):
		#computation of the barycenter (moment 1 of image) on ZOI using OpenCV
		#white_spot must be True if spots are white on a dark material
		# The median filter helps a lot for real life images ...
		while True:
			try:
				image,minx,miny=recv_.recv()[:]
				if self.update_tresh:
					self.thresh=threshold_otsu(image)
				bw=cv2.medianBlur(image,5)>self.thresh
				#print "thresh : ", self.thresh
				if not (self.white_spot):
					bw=1-bw
				M = cv2.moments(bw*255.)
				Px=M['m01']/M['m00']
				Py=M['m10']/M['m00']
				if self.NumOfReg==1:
					a=M['mu20']/M['m00']
					b=-M['mu11']/M['m00']
					c=M['mu02']/M['m00']
					#print "a,c : ", a,c
					l1=0.5*((a+c)+np.sqrt(4*b**2+(a-c)**2))
					l2=0.5*((a+c)-np.sqrt(4*b**2+(a-c)**2))
					minor_axis=4*np.sqrt(l2)
					major_axis=4*np.sqrt(l1)
					if (a-c)==0:
						if b>0:
							theta=-np.pi/4
						else:
							theta=np.pi/4
					else:
						theta=0.5*np.arctan2(2*b,(a-c))
					#print "min,maj,theta :" ,minor_axis,major_axis,theta
					Dx=max(np.abs(major_axis*np.cos(theta)),np.abs(minor_axis*np.sin(theta)))
					Dy=max(np.abs(major_axis*np.sin(theta)),np.abs(minor_axis*np.cos(theta)))
					Px=Dx
					Py=Dy
					#print "Dx,Dy : ", Dx,Dy

				else: #if 2 or 4 spots
					# we add minx and miny to go back to global coordinate:
					Px+=minx
					Py+=miny
				miny_, minx_, h, w= cv2.boundingRect((bw*255).astype(np.uint8)) # cv2 returns x,y,w,h but x and y are inverted
				maxy_=miny_+h
				maxx_=minx_+w
				minx=minx-self.border+minx_
				miny=miny-self.border+miny_
				maxx=minx+self.border+maxx_
				maxy=miny+self.border+maxy_
				recv_.send([Px,Py,minx,miny,maxx,maxy])
			except (Exception,KeyboardInterrupt) as e:
				print "Exception in barycenter : ",e
				try:
					recv_.send(["Error"]) # kill pill
				except:
					pass
				raise

	def main(self):
		#self.cap = cv2.VideoCapture(cv2.CAP_XIAPI + 1)
		#print "cam 2 live!"
		#self.cap.set(cv2.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
		#self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,2048);  # doesn't work for this one
		#self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2048); # reducing this one allows one to increase the FPS
		###self.cap.set(cv2.CAP_PROP_XI_OFFSET_Y,300); # Vertical axis
		###self.cap.set(cv2.CAP_PROP_XI_OFFSET_X,560); # horizontal axis from the left
		#self.cap.set(cv2.CAP_PROP_EXPOSURE,8000) # setting up exposure
		#self.cap.set(cv2.CAP_PROP_GAIN,0) #setting up gain
		#ret, frame = self.cap.read()
		#ret, frame = self.cap.read()
		#Array=pd.DataFrame([[time.time()-self.t0,self.L0x,self.L0y,0,0]],columns=self.labels)
		Array=OrderedDict(zip(self.labels,[time.time()-self.t0,self.L0x,self.L0y,0,0]))
		#t3_=time.time()
		#t3+=t3_-t2_
		try:
			for output in self.outputs:
				output.send(Array)
		except TimeoutError:
			raise
		except AttributeError: #if no outputs
			pass
			
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
		image = self.camera.sensor.getImage() # eliminate the first frame, most likely corrupted
		#self.t0 = time.time()
		while True:
			try:	
				t2=time.time()
				image = self.camera.sensor.getImage() # read a frame
				#if j%100==0 and j>0: # this loop is for test only
					#ret, frame = self.cap.read()
					#t2=time.time()
					#print "time diff : ", t2-t1
				#image1=sitk.GetImageFromArray(image)
				#sitk.WriteImage(image1,"/home/corentin/Bureau/img_to_delete/img_videoExtenso%.5d.tiff" %j)
				
					#image2=sitk.GetImageFromArray(frame)
					#sitk.WriteImage(image2,self.save_directory+"img_mouchetis%.5d.tiff" %j)
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
					send_[i].send([image[int(self.minx[i])-1:int(self.maxx[i])+1,int(self.miny[i])-1:int(self.maxy[i])+1],self.minx[i]-1,self.miny[i]-1])
				for i in range(0,self.NumOfReg):
					self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=send_[i].recv()[:] #self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]
					#self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=self.barycenter_opencv(image[self.minx[i]:self.maxx[i],self.miny[i]:self.maxy[i]],self.minx[i],self.miny[i])
				
				minx_=self.minx.min()
				miny_=self.miny.min()
				maxx_=self.maxx.max()
				maxy_=self.maxy.max()		
				if self.NumOfReg ==4 or self.NumOfReg ==2:
					Lx=self.Points_coordinates[:,0].max()-self.Points_coordinates[:,0].min()
					Ly=self.Points_coordinates[:,1].max()-self.Points_coordinates[:,1].min()
					Dx=100.*((Lx)/self.L0x-1.)
					Dy=100.*((Ly)/self.L0y-1.)
				elif self.NumOfReg ==1:
					#print self.Points_coordinates
					Ly=self.Points_coordinates[0,0]
					Lx=self.Points_coordinates[0,1]
					Dy=100.*((Ly)/self.L0y-1.)
					Dx=100.*((Lx)/self.L0x-1.)
				self.Points_coordinates[:,1]-=miny_
				self.Points_coordinates[:,0]-=minx_
				
				#Array=pd.DataFrame([[time.time()-self.t0,Lx,Ly,Dx,Dy]],columns=self.labels)
				Array=OrderedDict(zip(self.labels,[time.time()-self.t0,Lx,Ly,Dx,Dy]))
				#print Array
				try:
					for output in self.outputs:
						output.send(Array)
				except TimeoutError:
					raise
				except AttributeError: #if no outputs
					pass	
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
				if j%100==0 and j>0:
					t_now=time.time()
					#print "FPS: ", 100/(t_now-last_ttimer)
					stdout.write("\rFPS: %2.2f" % (100/(t_now-last_ttimer)))
					stdout.flush()
					#t2=time.time()
					#print "time diff : ", t2-t1
					#image1=sitk.GetImageFromArray(image)
					#sitk.WriteImage(image1,self.save_directory+"img_videoExtenso%.5d_t%3.3f_Exx%2.2f_Eyy%2.2f.tiff" %(j,(t2-self.t0),Lx,Ly))
					#image2=sitk.GetImageFromArray(frame)
					#sitk.WriteImage(image2,self.save_directory+"img_mouchetis%.5d_t%3.3f_Exx%2.2d_Eyy%2.2f.tiff" %(j,(t2-self.t0),Lx,Ly))
					last_ttimer=t_now
				
				j+=1
			except ValueError: # if lost spots in barycenter
				try:
					song_list=glob.glob('/home/*.wav')
					song = pyglet.media.load(random.choice(song_list))
					song.play()
					pyglet.clock.schedule_once(lambda x:pyglet.app.exit(), 10) # stop music after 10 sec
					pyglet.app.run()
				except:
					pass
				raise Exception("Spots lost")
			except (Exception,KeyboardInterrupt) as e:
				print "Exception in videoextenso : ",e
				proc.terminate()
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