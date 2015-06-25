from multiprocessing import Process
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import struct
np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd


class MasterBlock(object):
	"""
Main class for block architecture. All blocks should inherit this class.
	
Methods:
--------
main() : override it to define the main function of this block.
add_input(Link object): add a Link object as input.
add_output(Link object) : add a Link as output.
start() : start the main() method as a Process. It is designed to catch all 
	exceptions/error and terminate the process for safety. It also 	re-raises 
	the exception/error for upper-level handling.
stop() : stops the process.
	"""
	instances = []
	def __new__(cls, *args, **kwargs): #	Keeps track of all instances
		instance = super(MasterBlock, cls).__new__(cls, *args, **kwargs)
		instance.instances.append(instance)
		return instance
	
	def main(self):
		pass
	
	def add_output(self,link):
		try: # test if the outputs list exist
			a_=self.outputs[0]
		except AttributeError: # if it doesn't exist, create it
			self.outputs=[]
		self.outputs.append(link)
			
	def add_input(self,link):
		try: # test if the outputs list exist
			a_=self.inputs[0]
		except AttributeError: # if it doesn't exist, create it
			self.inputs=[]
		self.inputs.append(link)
	
	def start(self):
		try:
			self.proc=Process(target=self.main,args=())
			self.proc.start()
		except:
			self.proc.terminate()
			raise #raise the error to the next level for global shutdown
		
	def stop(self):
		self.proc.terminate()


class PathGenerator(MasterBlock):
	"""Many to one block. Compactate several data streams into arrays."""
	def __init__(self,t0,send_freq=1000,actuator=None,waveform=["sinus"],freq=[None],time_cycles=[None],amplitude=[1],offset=[0],phase=[],init=0):
		"""
Compacter(acquisition_step)

Read data inputs and save them in an array of length acquisition_step.
This block must be used to send data to the Saver or the Grapher.
Input values sent by the Links must be array (1D).
If you have multiple data input from several streamers, use multiple Compacter.
You should use several input only if you know that they have the same frequency.
You can have multiple outputs.

Parameters
----------
acquisition_step : int
	Number of values to save in each data-stream before returning the array.
	
Returns:
--------
Numpy array of shape (number_of_values_in_input,acquisition_step)

		"""
		print "PathGenerator!"
		self.nb_step=len(waveform)
		self.t0=t0
		self.send_freq=send_freq
		self.actuator=actuator
		self.waveform=waveform
		self.freq=freq
		self.time_cycles=time_cycles
		self.amplitude=amplitude
		self.offset=offset
		self.phase=phase
		self.init=init
		self.alpha=self.init
		self.step=0
	def main(self):
		self.labels=['t','signal']
		last_t=self.t0
		#i=0
		t_step=self.t0
		while self.step<self.nb_step:
			t_add=self.phase[self.step]/(2*np.pi*self.freq[self.step])
			while (time.time()-t_step)<self.time_cycles[self.step] or self.time_cycles==[None]:
				while time.time()-last_t<1./self.send_freq:
					time.sleep(1./(100*self.send_freq))
				last_t=time.time()
				t=last_t+t_add
				if self.waveform[self.step]=="sinus":
					self.alpha=self.amplitude[self.step]*np.sin(2*np.pi*(t-self.t0)*self.freq[self.step])+self.offset[self.step]
				elif self.waveform[self.step]=="triangle":
					self.alpha=(4*self.amplitude[self.step]*self.freq[self.step])*((t-self.t0)-(np.floor(2*self.freq[self.step]*(t-self.t0)+0.5))/(2*self.freq[self.step]))*(-1)**(np.floor(2*self.freq[self.step]*(t-self.t0)+0.5))+self.offset[self.step]
				elif self.waveform[self.step]=="square":
					self.alpha=self.amplitude[self.step]*np.sign(np.cos(2*np.pi*(t-self.t0)*
								self.freq[self.step]))+self.offset[self.step]
				else:
					raise Exception("invalid waveform : use sinus,triangle or square")
				Array=pd.DataFrame([t-self.t0,self.alpha],self.labels)
				t_,cmd_=self.actuator.set_cmd(self.alpha)
				#if self.cycles!=[None] and int((t-t_step)*self.freq[self.step])>(i):
					#i+=1
				try:
					for output in self.outputs:
						output.send(Array)
				except:
					pass
			self.step+=1
			t_step=time.time()
			#i=0
			


class CameraDisplayer(MasterBlock):
	"""Many to one block. Compactate several data streams into arrays."""
	def __init__(self):
		"""
Compacter(acquisition_step)

Read data inputs and save them in an array of length acquisition_step.
This block must be used to send data to the Saver or the Grapher.
Input values sent by the Links must be array (1D).
If you have multiple data input from several streamers, use multiple Compacter.
You should use several input only if you know that they have the same frequency.
You can have multiple outputs.

Parameters
----------
acquisition_step : int
	Number of values to save in each data-stream before returning the array.
	
Returns:
--------
Numpy array of shape (number_of_values_in_input,acquisition_step)

		"""
		print "cameraDisplayer!"
      
	def main(self):
		plt.ion()
		fig=plt.figure()
		ax=fig.add_subplot(111)
		first_loop=True
		while True:
			#print "top loop"
			frame=self.inputs[0].recv()
			if frame != None:
				#print frame[0][0]
				if first_loop:
					im = plt.imshow(frame,cmap='gray')
					first_loop=False
				else:
					im.set_array(frame)
				plt.draw()



class Compacter(MasterBlock):
	"""Many to one block. Compactate several data streams into arrays."""
	def __init__(self,acquisition_step):
		"""
Compacter(acquisition_step)

Read data inputs and save them in an array of length acquisition_step.
This block must be used to send data to the Saver or the Grapher.
Input values sent by the Links must be array (1D).
If you have multiple data input from several streamers, use multiple Compacter.
You should use several input only if you know that they have the same frequency.
You can have multiple outputs.

Parameters
----------
acquisition_step : int
	Number of values to save in each data-stream before returning the array.
	
Returns:
--------
Numpy array of shape (number_of_values_in_input,acquisition_step)

		"""
		print "compacter!"
		self.acquisition_step=acquisition_step
      
	def main(self):
		while True:
			data=[0 for x in xrange(self.acquisition_step)]
			for i in range(self.acquisition_step):
				if i==0:
					Data=self.inputs[0].recv()
				else:
					Data1=self.inputs[0].recv()
				if len(self.inputs)!=1:
					for k in range(1,len(self.inputs)):
						data_recv=self.inputs[k].recv()
						if i ==0:
							Data=pd.concat([Data,data_recv])
						else:
							Data1=pd.concat([Data1,data_recv])
				if i!=0:
					Data=pd.concat([Data,Data1],axis=1)
			for j in range(len(self.outputs)):
				self.outputs[j].send(Data)


class Grapher(MasterBlock):
	"""Plot the input data"""
	def __init__(self,mode,*args):
		"""
Grapher(mode,*args)

The grapher receive data from the Compacter (via a Link) and plots it.

Parameters
----------
mode : string
	"dynamic" : create a dynamic graphe that updates in real time. 
	"static" : create a graphe that add new values at every refresh. If there 
	is too many data (> 20000), delete one out of 2 to avoid memory overflow.
args : tuple
	tuples of the columns index of input data for plotting. You can add as
	much as you want, depending on your computer performances.

Examples:
---------
graph=Grapher("dynamic",(0,1),(0,2))
	plot a dynamic graph with two lines plot( data[1]=f(data[0]) and
	data[2]=f(data[0]))
		"""
		print "grapher!"
		self.mode=mode
		self.args=args
		self.nbr_graphs=len(args)
      
	def main(self):
		print "main grapher"
		if self.mode=="dynamic":
			save_number=0
			fig=plt.figure()
			ax=fig.add_subplot(111)
			for i in range(self.nbr_graphs):	# init lines
				if i ==0:
					li = ax.plot(np.arange(1),np.zeros(1))
				else:
					li.extend(ax.plot(np.arange(1),np.zeros(1)))
			plt.grid()
			fig.canvas.draw()	# draw and show it
			plt.show(block=False)
			while True:
				Data=self.inputs[0].recv()	# recv data
				data=Data.values
				legend_=[Data.index[self.args[i][1]] for i in range(self.nbr_graphs)]
				if save_number>0: # lose the first round of data    
					if save_number==1: # init
						var=data
						plt.legend(legend_,bbox_to_anchor=(0., 1.02, 1., .102),
				 loc=3, ncol=len(legend_), mode="expand", borderaxespad=0.)
					elif save_number<=10:	# stack values
						var=np.hstack((var,data))
					else :	# delete old value and add new ones
						var=np.hstack((var[::,np.shape(data)[1]::],data))
					for i in range(self.nbr_graphs):	# update lines
						li[i].set_xdata(var[self.args[i][0]])
						li[i].set_ydata(var[self.args[i][1]])
				ax.relim()
				ax.autoscale_view(True,True,True)
				fig.canvas.draw() 
				if save_number <=10 :
					save_number+=1
					
		if self.mode=="static":
			plt.ion()
			fig=plt.figure()
			ax=fig.add_subplot(111)
			first_round=True
			k=[0]*self.nbr_graphs	# internal value for downsampling
			while True :
				Data=self.inputs[0].recv()	# recv data
				data=Data.values
				legend_=[Data.index[self.args[i][1]] for i in range(self.nbr_graphs)]
				if first_round:	# init at first round
					for i in range(self.nbr_graphs):
						if i==0:
							li=ax.plot(
								data[self.args[i][0]],data[self.args[i][1]],
								label='line '+str(i))
						else:
							li.extend(ax.plot(
								data[self.args[i][0]],data[self.args[i][1]],
								label='line '+str(i)))
					# Display legend on first round
					#legend_=['line '+str(i) for i in range(self.nbr_graphs)]
					plt.legend(legend_,bbox_to_anchor=(0., 1.02, 1., .102),
						loc=3,ncol=len(legend_), mode="expand",
						borderaxespad=0.)
					plt.grid()
					fig.canvas.draw()
					first_round=False
				else:	# not first round anymore
					for i in range(self.nbr_graphs):
						data_x=li[i].get_xdata()
						data_y=li[i].get_ydata()
						if len(data_x)>=20000:
							# if more than 20000 values, cut half
							k[i]+=1
							li[i].set_xdata(np.hstack((data_x[::2],
								  data[self.args[i][0]][::2**k[i]])))
							li[i].set_ydata(np.hstack((data_y[::2],
								  data[self.args[i][1]][::2**k[i]])))
						else:
							li[i].set_xdata(np.hstack((data_x,
								  data[self.args[i][0]][::2**k[i]])))
							li[i].set_ydata(np.hstack((data_y,
								  data[self.args[i][1]][::2**k[i]])))
				ax.relim()
				ax.autoscale_view(True,True,True)
				fig.canvas.draw() 


class MeasureAgilent34420A(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,t0,agilentSensor,labels=['t','R'],freq=None):
		"""
MeasureAgilent34420A(t0,agilentSensor,labels=['t','R'],freq=None)

This block read the value of the resistance measured by agilent34420A and send
the values through a Link object.
It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters:
-----------
t0 : float
	Time origin, common to every Blocks
agilentSensor : agilentSensor object
	See sensor.agilentSensor documentation.
labels : list
	The labels you want with your data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition device capability.
		"""
		self.t0=t0
		self.agilentSensor=agilentSensor
		self.labels=labels
		self.freq=freq

	def main(self):
		try:
			_a=self.inputs[:]
			trigger="external"
		except AttributeError:
			trigger="internal"
		timer=time.time()
		try:
			while True:
				data=[]
				if trigger=="internal":
					if self.freq!=None:
						while time.time()-timer< 1./self.freq:
							pass
					timer=time.time()
					data=[timer-self.t0]
					ret=self.agilentSensor.getData()
					if ret != False:
						data.append(ret)	
				if trigger=="external":
					if self.inputs.input_.recv(): # wait for a signal
						data=[time.time()-self.t0]
						ret=self.agilentSensor.getData()
						if ret != False:
							data.append(ret)	
				Array=pd.DataFrame(data,self.labels)
				for output in self.outputs:
					output.send(Array)

		except (KeyboardInterrupt):	
			self.agilentSensor.close()

class MeasureComediByStep(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,t0,comediSensor,labels=None,freq=None):
		"""
MeasureComediByStep(t0,comediSensor,labels=None,freq=None)

This streamer read the value on all channels ONE BY ONE and send the 
values through a Link object. it is slower than StreamerComedi, but works on 
every USB driver. 
It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters:
-----------
t0 : float
	Time origin, common to every Blocks
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list
	The labels you want with your data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition card capability.
		"""
		self.t0=t0
		self.comediSensor=comediSensor
		self.labels=labels
		self.freq=freq

	def main(self):
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
					data=[time.time()-self.t0]
					for channel_number in range(self.comediSensor.nchans):
						t,value=self.comediSensor.getData(channel_number)
						data.append(value)
				if trigger=="external":
					if self.inputs.input_.recv(): # wait for a signal
						data=[time.time()-self.t0]
					for channel_number in range(self.comediSensor.nchans):
						t,value=self.comediSensor.getData(channel_number)
						data.append(value)

				if self.labels==None:
					self.Labels=[i for i in range(self.comediSensor.nchans+1)]
				Array=pd.DataFrame(data,self.labels)
				for output in self.outputs:
					output.send(Array)

		except (KeyboardInterrupt):	
			self.comediSensor.close()
			


class Reader(MasterBlock):
	"""
Children class of MasterBlock. Read and print the input Link.
	"""
	
	
	def __init__(self,k):
		"""
(Reader(k))

Create a reader that prints k and the input data in continuous.

Parameters:
-----------
k : printable (int or float or string)
	Some identifier for this particular instance of Reader
	
		"""
		#super(Reader, self).__init__()	
		self.k=k  
		
	def main(self):
		while True:
			for input_ in self.inputs:
				self.data=input_.recv()
			print self.k,self.data
			

class Saver(MasterBlock):
	"""Saves data in a file"""
	def __init__(self,log_file):
		"""
Saver(log_file)

Saves data in a file. Be aware that the log file needs to be cleaned before 
starting this function, otherwise it just keep writing a the end of the file.

Parameters
----------
log_file : string
	Path to the log file. If non-existant, will be created.

		"""
		#super(Saver, self).__init__()	
		print "saver!"
		self.log_file=log_file
		if not os.path.exists(os.path.dirname(self.log_file)):
			# check if the directory exists, otherwise create it
			os.makedirs(os.path.dirname(self.log_file))
      
	def main(self):
		first=True
		while True:
			#data=self.inputs[0].recv()
			Data=self.inputs[0].recv()	# recv data
			data=Data.values
			legend_=Data.index
			fo=open(self.log_file,"a")		# "a" for appending
			fo.seek(0,2)		#place the "cursor" at the end of the file
			if first:
				fo.write(str([legend_[i] for i in range(len(legend_))])+"\n")
				first =False
			data_to_save=str(np.transpose(data))+"\n"
			fo.write(data_to_save)
			fo.close()


  
class Streamer(MasterBlock):
	"""
Children class of MasterBlock. Send a fake stream of data in a pipe.
	"""
	
	
	def __init__(self,t0):
		"""
Send iterated value through a Link object.

Parameters:
-----------
t0 : float
	Time origin, common to every Blocks
		"""
		self.t0=t0
		
	def main(self):
		self.i=0
		while True:
			time.sleep(0.001)
			for output in self.outputs:
				output.send([time.time()-self.t0,self.i])
			self.i+=1     


class StreamerComedi(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,t0,comediSensor,labels=None,freq=8000):
		"""
This streamer read the value on all channels at the same time and send the 
values through a Link object. It can be very fast, but needs need an USB 2.0
port drove by ehci to work properly. xhci driver DOES NOT work (for now).

Parameters:
-----------
t0 : float
	Time origin, common to every Blocks
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list
	The labels you want with your data.
freq : int (default 8000)
	the frequency you need.
		"""
		import comedi as c
		self.labels=labels
		self.c=c
		self.t0=t0
		self.comediSensor=comediSensor
		
		self.fd = self.c.comedi_fileno(self.comediSensor.device)	# get a file-descriptor

		self.BUFSZ = 10000	# buffer size
		self.freq=freq	# acquisition frequency
	
		self.nchans = len(self.comediSensor.channels)	# number of channels
		self.aref =[self.c.AREF_GROUND]*self.nchans

		mylist = self.c.chanlist(self.nchans)	# create a chanlist of length nchans
		self.maxdata=[0]*(self.nchans)
		self.range_ds=[0]*(self.nchans)

		for index in range(self.nchans):	# pack informations into the chanlist
			mylist[index]=self.c.cr_pack(self.comediSensor.channels[index],
						   self.comediSensor.range_num[index],
						   self.aref[index])
			self.maxdata[index]=self.c.comedi_get_maxdata(self.comediSensor.device,
									   self.comediSensor.subdevice,
									   self.comediSensor.channels[index])
			self.range_ds[index]=self.c.comedi_get_range(self.comediSensor.device,
									  self.comediSensor.subdevice,
									  self.comediSensor.channels[index],
									  self.comediSensor.range_num[index])

		cmd = self.c.comedi_cmd_struct()

		period = int(1.0e9/self.freq)	# in nanoseconds
		ret = self.c.comedi_get_cmd_generic_timed(self.comediSensor.device,
									   self.comediSensor.subdevice,
									   cmd,self.nchans,period)
		if ret: raise Exception("Error comedi_get_cmd_generic failed")
			
		cmd.chanlist = mylist # adjust for our particular context
		cmd.chanlist_len = self.nchans
		cmd.scan_end_arg = self.nchans
		cmd.stop_arg=0
		cmd.stop_src=self.c.TRIG_NONE

		ret = self.c.comedi_command(self.comediSensor.device,cmd)
		if ret !=0: raise Exception("comedi_command failed...")

	#Lines below are for initializing the format, depending on the comedi-card.
		data = os.read(self.fd,self.BUFSZ) # read buffer and returns binary data
		self.data_length=len(data)
		if self.maxdata[0]<=65536: # case for usb-dux-D
			n = self.data_length/2 # 2 bytes per 'H'
			self.format = `n`+'H'
		elif self.maxdata[0]>65536: #case for usb-dux-sigma
			n = self.data_length/4 # 2 bytes per 'H'
			self.format = `n`+'I'
			
	# init is over, start acquisition and stream
	def main(self):
		try:
			while True:
				array=np.zeros(self.nchans+1)
				data = os.read(self.fd,self.BUFSZ) # read buffer and returns binary
				if len(data)==self.data_length:
					datastr = struct.unpack(self.format,data)
					if len(datastr)==self.nchans: #if data not corrupted
						array[0]=time.time()-self.t0
						for i in range(self.nchans):
							array[i+1]=self.c.comedi_to_phys((datastr[i]),
												self.range_ds[i],
												self.maxdata[i])
						if self.labels==None:
							self.Labels=[i for i in range(self.nchans+1)]
						Array=pd.DataFrame(array,self.labels)
						for output in self.outputs:
							output.send(Array)

		except (KeyboardInterrupt):	
			self.comediSensor.close()


class StreamerCamera(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,Camera,freq=None,save=False,
			  save_directory="./images/"):
		"""
StreamerCamera(cameraSensor,freq=None,save=False,save_directory="./images/")

This block fetch images from a camera object, save and/or transmit them to 
another block. It can be triggered by a Link sending boolean or internally 
by defining the frequency.

Parameters:
-----------
cameraSensor : cameraSensor object
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
		self.Camera=Camera
		self.sitk = sitk
		#self.cameraSensor=cameraSensor
		#self.cameraSensor=Ximea()
		self.freq=freq
		self.save=save
		self.i=0
		self.save_directory=save_directory
		if not os.path.exists(self.save_directory) and self.save:
			os.makedirs(self.save_directory)

	def main(self):
		self.cameraSensor=self.Camera()
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
