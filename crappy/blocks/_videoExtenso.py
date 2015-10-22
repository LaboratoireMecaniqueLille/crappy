from ._meta import MasterBlock
from multiprocessing import Process, Pipe
import numpy as np
np.set_printoptions(threshold='nan', linewidth=500)
import time
import pandas as pd
import cv2
from ..technical import TechnicalCamera as tc
import SimpleITK as sitk # only for testing
from skimage.filter import threshold_otsu, rank
from skimage.measure import regionprops
from skimage.morphology import label,erosion, square,dilation
from skimage.segmentation import clear_border
try:
	import pyglet
	import glob
	import random
except ImportError:
	print "no sound module installed"
	

class VideoExtenso(MasterBlock): 
	"""
This class detects 4 spots, and evaluate the deformations Exx and Eyy.
	"""
	def __init__(self,camera="ximea",numdevice=0,xoffset=0,yoffset=0,width=2048,height=2048,white_spot=True,display=True,labels=['t(s)','Lx','Ly','Exx(%)','Eyy(%)']):
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
		self.numdevice=numdevice
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
			self.Points_coordinates=self.camera.Points_coordinates
			self.width=self.camera.width
			self.height=self.camera.height
			self.xoffset=self.camera.xoffset
			self.yoffset=self.camera.yoffset
			self.exposure=self.camera.exposure
			self.gain=self.camera.gain
			if self.NumOfReg==4 or self.NumOfReg==2 or self.NumOfReg==1: 
				go=True
			else:	#	If detection goes wrong, start again
				print " Spots detected : ", self.NumOfReg	
			# following is for tests only
			#self.save_directory="/home/biaxe/Bureau/Publi/"
			#fo=open(self.save_directory+"L0.txt","a")		# "a" for appending
			#fo.seek(0,2)		#place the "cursor" at the end of the file
			#data_to_save="L0x : "+str(self.L0x)+"\n"+"Loy : "+str(self.L0y)
			#fo.write(data_to_save)
			#fo.close()

	#def barycenter_monospot(self,recv_):
		
		#img = cv2.imread('star.jpg',0)
		#ret,thresh = cv2.threshold(img,127,255,0)
		#contours,hierarchy = cv2.findContours(thresh, 1, 2)
		#cnt = contours[0]
		#x,y,w,h = cv2.boundingRect(cnt)
		#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		
		
		
	def barycenter_opencv(self,recv_):
		"""
		computation of the barycenter (moment 1 of image) on ZOI using OpenCV
		white_spot must be True if spots are white on a dark material
		"""
		# The median filter helps a lot for real life images ...
		while True:
			try:
				#last_region_area=0
				image,minx,miny=recv_.recv()[:]
				#print "shape :",  image.shape
				#image=rank.median(image,square(15))
				self.thresh=threshold_otsu(image)
				bw=cv2.medianBlur(image,5)>self.thresh
				if not (self.white_spot):
					bw=1-bw
				M = cv2.moments(bw*255.)
				Px=M['m01']/M['m00']
				Py=M['m10']/M['m00']
				if self.NumOfReg==1:
					#bw = dilation(bw,square(3))
					#bw = erosion(bw,square(3))
					#cleared = bw.copy()
					#clear_border(cleared)
					#label_image = label(cleared)
					#borders = np.logical_xor(bw, cleared)
					#label_image[borders] = -1
					#print "NofReg :", len(regionprops(label_image))
					#for region in regionprops(label_image):
						#if region.area>last_region_area:
							#minx_, miny_, maxx_, maxy_ = region.bbox
							#Px=minx+(minx_+ maxx_)/2.
							#Py=minx+(miny_+ maxy_)/2.
							#last_region_area=region.area
					#thresh = (bw*255).astype(np.uint8)
					#print "1"
					#ret,thresh = cv2.threshold(image,self.thresh,255,0)
					#print "2"
					#image,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
					#cnt = contours[0]
					#print contours
					#ellipse = cv2.fitEllipse(cnt)a
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
					
					#print Dx, Dy, Px,Py
						#Px=M['m01']/M['m00']
						#Py=M['m10']/M['m00'] 
					#maxy_=Dx #Py+Dy/2.
					#minx_= 0 #Px-Dx/2.
					#miny_=0 #Py-Dy/2.
					#maxx_=Dy #Px+Dx/2.
					Px=Dx
					Py=Dy
					print "Dx,Dy : ", Dx,Dy
						#if minx_<0:
							#minx_=0
						#if miny_<0:
							#miny_=0
					#print "ellipse: ", minx_, maxx_, miny_, maxy_ 
					#Px=minx+(minx_+ maxx_)/2.
					#Py=minx+(miny_+ maxy_)/2.
				else: 
					# we add minx and miny to go back to global coordinate:
					Px+=minx
					Py+=miny
				miny_, minx_, h, w= cv2.boundingRect((bw*255).astype(np.uint8)) # cv2 returns x,y,w,h but x and y are inverted
				maxy_=miny_+h
				maxx_=minx_+w
					#print "rect : ",minx_,miny_,maxx_,maxy_
					# Determination of the new bounding box using global coordinates and the margin
				minx=minx-self.border+minx_
				miny=miny-self.border+miny_
				maxx=minx+self.border+maxx_
				maxy=miny+self.border+maxy_
				#if self.NumOfReg==1:
					#maxx=Dx
					#maxy=Dy
				recv_.send([Px,Py,minx,miny,maxx,maxy])
			except (Exception,KeyboardInterrupt) as e:
				print "Exception in barycenter : ",e
				try:
					recv_.send(["Error"]) # kill pill
				except:
					pass
				raise

	def main(self):
		"""
		main function, command the videoextenso and the motors
		"""
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
		Array=pd.DataFrame([[time.time()-self.t0,self.L0x,self.L0y,0,0]],columns=self.labels)
		#t3_=time.time()
		#t3+=t3_-t2_
		try:
			for output in self.outputs:
				output.send(Array)
		except AttributeError:
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
					#sitk.WriteImage(image1,self.save_directory+"img_videoExtenso%.5d.tiff" %j)
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
				Array=pd.DataFrame([[time.time()-self.t0,Lx,Ly,Dx,Dy]],columns=self.labels)
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
				if j%100==0 and j>0:
					t_now=time.time()
					print "FPS: ", 100/(t_now-last_ttimer)
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