import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import SimpleITK as sitk
from matplotlib.widgets import Slider, Button
rcParams['font.family'] = 'serif'
#ps aux | grep python 					 # KILL python process ...
#kill -9 insert_here_the_python_thread_number		 # ... try it if ximea won't open again.
plt.close('all')

############################## Parameters : All you need to modify is here
nbr_images=None # enter here the number of images you need to save when clicking the REC button. If None, take images until stopped by CTRL+C
save_directory="/home/essais-2015-1/Bureau/essai_tribo/" # path to save repository. BE AWARE that this scripts will erase previous images without regrets or remorse.
exposure= 10000 # exposition time, in microseconds
gain=0.5
height=1024 # reducing this one allows one to increase the FPS
width=2048 # doesn't work for this one
offset_v=800
offset_h=0
data_format=0 #0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW	
external_trigger= True #set to True if you trig with external source (arduino...). BE AWARE there is a 10s waiting time for the ximea, meaning if you wait more that 10 sec to trigg, ximea will return an error and stop working.
set_FPS=False # set to True if you want to manually set the frame rate. It has 0.1 FPS precison @88FPS . If you need more precision, please use external trigger with arduino.
FPS=50 # set here the frame rate you need. This parameter will only work if set_FPS =True.
numdevice = 0 # Set the number of the camera (if several cameras plugged)
window=False # if True, open another window to keep displaying the camera images while saving them (when REC button clicked). BE AWARE that it slows the script to ~5 FPS !
##############################

#cap = cv2.VideoCapture(cv2.CAP_XIAPI) # open the ximea device
cap = cv2.VideoCapture(cv2.CAP_XIAPI + numdevice) # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1 

if external_trigger==True:	# this condition activate the trigger mode
	cap.set(cv2.CAP_PROP_XI_TRG_SOURCE,1)
	cap.set(cv2.CAP_PROP_XI_GPI_SELECTOR,1)
	cap.set(cv2.CAP_PROP_XI_GPI_MODE,1)

cap.set(cv2.CAP_PROP_XI_DATA_FORMAT,data_format) #0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW	

if data_format ==1 or data_format==6: #increase the FPS in 10 bits
	cap.set(cv2.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH,10)
	cap.set(cv2.CAP_PROP_XI_DATA_PACKING,1)


cap.set(cv2.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);	# doesn't work for this one
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); # reducing this one allows one to increase the FPS
cap.set(cv2.CAP_PROP_XI_OFFSET_Y,offset_v); # Vertical axis
cap.set(cv2.CAP_PROP_XI_OFFSET_X,offset_h); # horizontal axis from the left
#cap.set(cv2.CAP_PROP_XI_DOWNSAMPLING,0) # activate this one if you need to downsample your images, i.e if you need a very high FPS and other options are not enough
#print cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#print cap.get(cv2.CAP_PROP_FRAME_HEIGHT); 

cap.set(cv2.CAP_PROP_EXPOSURE,exposure) # setting up exposure
cap.set(cv2.CAP_PROP_GAIN,gain) #setting up gain
ret, frame = cap.read() # read a frame
if ret:
	pass
else:
	frame=np.zeros((height,width))
	frame[0,0]=1.
	

### initialising the histogram
if cap.get(cv2.CAP_PROP_XI_DATA_FORMAT)==0 or cap.get(cv2.CAP_PROP_XI_DATA_FORMAT)==5:
	x=np.arange(0,256,4)
if cap.get(cv2.CAP_PROP_XI_DATA_FORMAT)==1 or cap.get(cv2.CAP_PROP_XI_DATA_FORMAT)==6:
	x=np.arange(0,1024,4)
hist=np.ones(np.shape(x))

### initialising graph and axes
rat = 0.7
Width=7
Height=7.
fig=plt.figure(figsize=(Height, Width))
axim = fig.add_axes([0.15, 0.135, rat, rat*(Height/Width)]) # Image frame
cax = fig.add_axes([0.17+rat, 0.135, 0.02, rat*(Height/Width)]) # colorbar frame
axhist=fig.add_axes([0.15,(0.17+rat),rat,0.1]) # histogram frame
axhist.set_xlim([0,max(x)]) #set histogram limit in x...
axhist.set_ylim([0,1]) # ... and y

im = axim.imshow(frame,cmap=plt.cm.gray,interpolation='nearest') # display the first image
li,= axhist.plot(x,hist) #plot first histogram
cb = fig.colorbar(im, cax=cax) #plot colorbar
cax.axis('off')


### define cursors here
axcolor = 'lightgoldenrodyellow'
axExp = plt.axes([0.15, 0.02,rat, 0.03], axisbg=axcolor) # define position and size
sExp = Slider(axExp, 'Exposure', 200, 50000, valinit=exposure) #Exposition max = 1000000 # define slider with previous position and size
axGain= plt.axes([0.15, 0.07,rat, 0.03], axisbg=axcolor)
sGain = Slider(axGain, 'Gain', -1, 6, valinit=gain)

def update(val): # this function updates the exposure and gain values 
	cap.set(cv2.CAP_PROP_EXPOSURE,sExp.val)
	cap.set(cv2.CAP_PROP_GAIN,sGain.val)
	fig.canvas.draw_idle()
	
sExp.on_changed(update) # call for update everytime the cursors change
sGain.on_changed(update)



### define buttons here
RECax = plt.axes([0.01, (0.15+rat)/2, 0.05, 0.05]) # define size and position
button = Button(RECax, 'REC', color='red', hovercolor='0.975') # define button

def REC(event): # when called, read "nbr_images" and save them as .tiff in save_directory
	t0=time.time()
	last_t=0
	i=0
	if nbr_images is None:
		while True:
			if set_FPS==True and last_t!=0: #This loop is used to set the FPS
				while (time.time()-last_t) < 1./FPS:
					indent=True
			last_t=time.time()
			ret, frame = cap.read()
			if ret:
				image=sitk.GetImageFromArray(frame)
				sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %i) ### works fast in 8 or 16 bit, always use sitk.
				i+=1
			else:
				pass
			#i+=1
	else:
		while(i<nbr_images):
			if set_FPS==True and last_t!=0: #This loop is used to set the FPS
				while (time.time()-last_t) < 1./FPS:
					indent=True
			last_t=time.time()
			ret, frame = cap.read()
			if ret:
				image=sitk.GetImageFromArray(frame)
				sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %i) ### works fast in 8 or 16 bit, always use sitk.
				i+=1
			else:
				pass
			#image=sitk.GetImageFromArray(frame)
			#sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %i) ### works fast in 8 or 16 bit, always use sitk.
			#i+=1
	t=time.time()-t0
	print "FPS = %s"%(nbr_images/t)
	

def REC_one(event): # when called, read 1 image and save it as .tiff in save_directory with a timestamp, so the next REC will not erase the previous one
	ret, frame = cap.read()
	if ret:
		image=sitk.GetImageFromArray(frame)
		sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %(time.time())) ### works fast in 8 or 16 bit, always use sitk.
	else:
		pass
	#image=sitk.GetImageFromArray(frame)
	#sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %(time.time())) ### works fast in 8 or 16 bit, always use sitk.

def REC2(event):
	i=0
	last_t=0
	t0=time.time()
	while True:
		if set_FPS==True and last_t!=0: #This loop is used to set the FPS
			while (time.time()-last_t) < 1./FPS:
				indent=True
		last_t=time.time()
		ret, frame = cap.read()
		if ret:
			image=sitk.GetImageFromArray(frame)
			sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %i) ### works fast in 8 or 16 bit, always use sitk.
			i+=1
		else:
			frame=np.zeros((height,width))
			frame[0,0]=1.
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		cv2.imshow('image',frame)
		#cv2.imwrite(save_directory + 'frame%5.5d.tiff'%i,frame)
		#image=sitk.GetImageFromArray(frame)
		#sitk.WriteImage(image,save_directory+"img_%.5d.tiff" %i) ### works fast in 8 or 16 bit, always use sitk.
		#i+=1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyWindow('image')
			t=time.time()-t0
			print "FPS = %s"%(i/t)
			break

if nbr_images== 1 and window==False:
	button.on_clicked(REC_one) # on click, call the REC function
elif nbr_images!=1 and window==False:
	button.on_clicked(REC)
elif window==True:
	button.on_clicked(REC2)


### Main
def get_frame(i):
	try:
		ret, frame = cap.read() # read a frame		
		if ret:
			print "ret ok"
			im.set_data(frame) #change previous image by new frame
			im.set_clim([frame.min(), frame.max()]) # re-arrange colorbar limits
			histogram=cv2.calcHist([frame],[0],None,[len(x)],[0,max(x)]) # evalute new histogram
			histogram=np.sqrt(histogram) # this operation aims to improve the histogram visibility (avoid flattening)
			li.set_ydata((histogram-histogram.min())/(histogram.max()-histogram.min())) # change previous histogram
		else:
			print "ret error"
			frame=np.zeros((height,width))
			frame[0,0]=1.
			im.set_data(frame) #change previous image by new frame
			im.set_clim([frame.min(), frame.max()]) # re-arrange colorbar limits
		if i == 1:
			cax.axis('on')
		#im.set_data(frame) #change previous image by new frame
		#im.set_clim([frame.min(), frame.max()]) # re-arrange colorbar limits
		#histogram=cv2.calcHist([frame],[0],None,[len(x)],[0,max(x)]) # evalute new histogram
		#histogram=np.sqrt(histogram) # this operation aims to improve the histogram visibility (avoid flattening)
		#li.set_ydata((histogram-histogram.min())/(histogram.max()-histogram.min())) # change previous histogram
	except TypeError:
		frame=np.zeros((height,width))
		frame[0,0]=1.
		im.set_data(frame)
	return cax, axim , axhist # return the values that need to be updated



ani = animation.FuncAnimation(fig, get_frame, interval=20, frames=20, blit=False) # This function call the get_frame function to update averything in the figure.
plt.show()