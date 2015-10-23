from multiprocessing import Process, Pipe
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd
import sys


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
		except Exception as e:
			print "Exception : ", e
			self.proc.terminate()
			raise #raise the error to the next level for global shutdown
		
	def join(self):
		self.proc.join()
		
	def stop(self):
		self.proc.terminate()
		
	def set_t0(self,t0):
		self.t0=t0


class CameraDisplayer(MasterBlock):
	"""Simple camera displayer. Must receive frames from StreamerCamera"""
	def __init__(self):
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

class CommandBiotens(MasterBlock):
	"""Receive a signal and translate it for the Biotens actuator"""
	def __init__(self, biotens_technicals, speed=5):
		"""
Receive a signal and translate it for the Biotens actuator.

CommandBiotens(biotens_technical,speed=5)

Parameters
----------
biotens_technicals : list of crappy.technical.Biotens object.

speed: int
		"""
		self.biotens_technicals=biotens_technicals
		self.speed=speed
		for biotens_technical in self.biotens_technicals:
			biotens_technical.actuator.clear_errors()
	
	def main(self):
		try:
			last_cmd=0
			while True:
				Data=self.inputs[0].recv()
				cmd=Data['signal'].values[0]
				if cmd!= last_cmd:
					for biotens_technical in self.biotens_technicals:
						biotens_technical.actuator.setmode_speed(cmd*self.speed)
					last_cmd=cmd
		except:
			for biotens_technical in self.biotens_technicals:
				biotens_technical.actuator.stop_motor()

class Compacter(MasterBlock):
	"""Many to one block. Compactate several data streams into arrays."""
	def __init__(self,acquisition_step):
		"""
Compacter(acquisition_step)

Read data inputs and save them in a panda dataframe of length acquisition_step.
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
Panda Dataframe of shape (number_of_values_in_input,acquisition_step)

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
							Data=pd.concat([Data,data_recv],axis=1)
						else:
							Data1=pd.concat([Data1,data_recv],axis=1)
				if i!=0:
					Data=pd.concat([Data,Data1])
			for j in range(len(self.outputs)):
				self.outputs[j].send(Data)
			#print Data

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
	tuples of the columns labels of input data for plotting. You can add as
	much as you want, depending on your computer performances.

Examples:
---------
graph=Grapher("dynamic",('t(s)','F(N)'),('t(s)','def(%)'))
	plot a dynamic graph with two lines plot( F=f(t) and def=f(t)
		"""
		print "grapher!"
		self.mode=mode
		self.args=args
		self.nbr_graphs=len(args)		
      
	def main(self):
		try:
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
					legend_=Data.columns[1:]
					if save_number>0: # lose the first round of data    
						if save_number==1: # init
							var=Data
							plt.legend(legend_,bbox_to_anchor=(0., 1.02, 1., .102),
					loc=3, ncol=len(legend_), mode="expand", borderaxespad=0.)
						elif save_number<=10:	# stack values
							var=pd.concat([var,Data])
						else :	# delete old value and add new ones
							var=pd.concat([var[np.shape(Data)[0]:],Data])
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
					legend_=Data.columns[1:]
					if first_round:	# init at first round
						for i in range(self.nbr_graphs):
							if i==0:
								li=ax.plot(
									Data[self.args[i][0]],Data[self.args[i][1]],
									label='line '+str(i))
							else:
								li.extend(ax.plot(
									Data[self.args[i][0]],Data[self.args[i][1]],
									label='line '+str(i)))
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
									Data[self.args[i][0]][::2**k[i]])))
								li[i].set_ydata(np.hstack((data_y[::2],
									Data[self.args[i][1]][::2**k[i]])))
							else:
								li[i].set_xdata(np.hstack((data_x,
									Data[self.args[i][0]][::2**k[i]])))
								li[i].set_ydata(np.hstack((data_y,
									Data[self.args[i][1]][::2**k[i]])))
					ax.relim()
					ax.autoscale_view(True,True,True)
					fig.canvas.draw() 
		except Exception as e:
			plt.close('all')


class MeasureAgilent34420A(MasterBlock):
	"""
Children class of MasterBlock. Send value through a Link object.
	"""
	def __init__(self,agilentSensor,labels=['t','R'],freq=None):
		"""
MeasureAgilent34420A(agilentSensor,labels=['t','R'],freq=None)

This block read the value of the resistance measured by agilent34420A and send
the values through a Link object.
It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters:
-----------
agilentSensor : agilentSensor object
	See sensor.agilentSensor documentation.
labels : list
	The labels you want with your data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition device capability.
		"""
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
				Array=pd.DataFrame([data],columns=self.labels)
				for output in self.outputs:
					output.send(Array)

		except (KeyboardInterrupt):	
			self.agilentSensor.close()

class MeasureComediByStep(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,comediSensor,labels=None,freq=None):
		"""
MeasureComediByStep(comediSensor,labels=None,freq=None)

This streamer read the value on all channels ONE BY ONE and send the 
values through a Link object. it is slower than StreamerComedi, but works on 
every USB driver. 
It can be triggered by a Link sending boolean (through "add_input" method),
or internally by defining the frequency.

Parameters:
-----------
comediSensor : comediSensor object
	See sensor.ComediSensor documentation.
labels : list
	The labels you want with your data.
freq : float or int, optional
	Wanted acquisition frequency. Cannot exceed acquisition card capability.
		"""
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
							time.sleep(1./(100*self.freq))
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
				Array=pd.DataFrame([data],columns=self.labels)
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
			fo=open(self.log_file,"a")		# "a" for appending
			fo.seek(0,2)		#place the "cursor" at the end of the file
			if first:
				legend_=Data.columns
				fo.write(str([legend_[i] for i in range(len(legend_))])+"\n")
				first =False
			data_to_save=str(data)+"\n"
			fo.write(data_to_save)
			fo.close()

class SignalGenerator(MasterBlock):
	"""Many to one block. Generate a signal."""
	def __init__(self,path=None,send_freq=800,repeat=False,labels=['t(s)','signal']):
		"""
SignalGenerator(path=None,send_freq=800,repeat=False,labels=['t(s)','signal'])

Calculate a signal, based on the time (from t0). There is several configurations,
see the examples section for more details.

Parameters
----------
path : list of dict
	Each dict must contain parameters for one step. See Examples section below.
	Available parameters are :
	* waveform : {'sinus','square','triangle','limit','hold'}
		Shape of your signal, for every step.
	* freq : int or float
		Frequency of your signal.
	* time : int or float or None
		Time before change of step, for every step. If None, means infinite.
	* cycles : int or float or None (default)
		Number of cycles before change of step, for every step. If None, means infinite.
	* amplitude : int or float
		Amplitude of your signal.
	* offset: int or float
		Offset of your signal.
	* phase: int or float
		Phase of your signal (in radians). If waveform='limit', phase will be 
		the direction of the signal (up or down).
	* lower_limit : [int or float,sting]
		Only for 'limit' mode. Define the lower limit as a value of the
		labeled signal : [value,'label']
	* upper_limit : [int or float,sting]
		Only for 'limit' mode. Define the upper limit as a value of the 
		labeled signal : [value,'label']
send_freq : int or float , default = 800
	Loop frequency. Use this parameter to avoid over-use of processor and avoid
	filling the link too fast.
repeat : Boolean, default=False
	Set True is you want to repeat your sequence forever.
labels : list of strings, default =['t(s)','signal']
	Allows you to set the labels of output data.

Returns:
--------
Panda Dataframe with time and signal. If waveform='limit', signal can be -1/0/1.

Examples:
---------
SignalGenerator(path=[{"waveform":"hold","time":3},
					{"waveform":"sinus","time":10,"phase":0,"amplitude":2,"offset":0.5,"freq":2.5},
					{"waveform":"triangle","time":10,"phase":np.pi,"amplitude":2,"offset":0.5,"freq":2.5},
					{"waveform":"square","time":10,"phase":0,"amplitude":2,"offset":0.5,"freq":2.5}
					{"waveform":"limit","cycles":3,"phase":0,"lower_limit":[-3,"signal"],"upper_limit":[2,"signal"]}],
					send_freq=400,repeat=True,labels=['t(s)','signal'])
In this example we displayed every possibility or waveform.
Every dict contains informations for one step.
The requiered informations depend on the type of waveform you need.

		"""
		print "PathGenerator!"
		self.path=path
		self.nb_step=len(path)
		self.send_freq=send_freq
		self.repeat=repeat
		self.labels=labels
		self.step=0
	def main(self):
		last_t=self.t0
		cycle=0
		first=True
		first_of_step=True
		t_step=self.t0
		Data=pd.DataFrame()
		while self.step<self.nb_step:
			current_step=self.path[self.step] 
			try:
				self.waveform=current_step["waveform"]
				if self.waveform=='hold':
					self.time=current_step["time"]
				elif self.waveform=='limit':
					self.cycles=current_step["cycles"]
					self.phase=current_step["phase"]
					self.lower_limit=current_step["lower_limit"]
					self.upper_limit=current_step["upper_limit"]
				else :
					self.time=current_step["time"]
					self.phase=current_step["phase"]
					self.amplitude=current_step["amplitude"]
					self.offset=current_step["offset"]
					self.freq=current_step["freq"]
					
			except KeyError as e:
				print "You didn't define parameter %s for step number %s" %(e,self.step)

				
			if self.waveform=="limit": #	 signal defined by a lower and upper limit
				alpha=np.sign(np.cos(self.phase))
				while self.cycles is None or cycle<self.cycles:
					while time.time()-last_t<1./self.send_freq:
						time.sleep(1./(100*self.send_freq))
					last_t=time.time()					
					###################################################get data
					for input_ in self.inputs:
						if input_.in_.poll() or first: # if there is data waiting
							Data=pd.concat([Data,input_.recv()],ignore_index=True)
					first=False
					last_upper = (Data[self.upper_limit[1]]).last_valid_index()
					last_lower=(Data[self.lower_limit[1]]).last_valid_index()
					first_lower=(Data[self.lower_limit[1]]).first_valid_index()
					first_upper=(Data[self.upper_limit[1]]).first_valid_index()
					if first_of_step:
						if alpha>0:
							if Data[self.upper_limit[1]][last_upper]>self.upper_limit[0]: # if value > high_limit
								alpha=-1
						elif alpha <0:
							if Data[self.lower_limit[1]][last_lower]<self.lower_limit[0]: # if value < low_limit
								alpha=1
						first_of_step=False
					if self.upper_limit==self.lower_limit: # if same limits
						alpha=0
						cycle=time.time()-t_step
					if alpha>0:
						if Data[self.upper_limit[1]][last_upper]>self.upper_limit[0]: # if value > high_limit
							alpha=-1
							cycle+=0.5
					elif alpha <0:
						if Data[self.lower_limit[1]][last_lower]<self.lower_limit[0]: # if value < low_limit
							alpha=1
							cycle+=0.5
					if last_upper!=first_upper and last_lower!=first_lower: # clean old data
						Data=Data[min(last_upper,last_lower):]
					Array=pd.DataFrame([[last_t-self.t0,alpha]],columns=self.labels)
					try:
						for output in self.outputs:
							output.send(Array)
					except:
						pass
				self.step+=1
				first_of_step=True
				cycle=0
				if self.repeat and self.step==self.nb_step:
					self.step=0
				t_step=time.time()
			elif self.waveform=="hold":
				while self.time is None or (time.time()-t_step)<self.time:
					while time.time()-last_t<1./self.send_freq:
						time.sleep(1./(100*self.send_freq))
					last_t=time.time()		
					if self.step==0:
						self.alpha=0
					else:
						if path[self.step-1]["waveform"]=="limit":
							alpha=0
						else:
							self.alpha=self.alpha
					Array=pd.DataFrame([[last_t-self.t0,self.alpha]],columns=self.labels)
					try:
						for output in self.outputs:
							output.send(Array)
					except:
						pass
				self.step+=1
				first_of_step=True
				cycle=0
				if self.repeat and self.step==self.nb_step:
					self.step=0
				t_step=time.time()
			else:
				t_add=self.phase/(2*np.pi*self.freq)
				while self.time is None or (time.time()-t_step)<self.time:
					while time.time()-last_t<1./self.send_freq:
						time.sleep(1./(100*self.send_freq))
					last_t=time.time()
					t=last_t+t_add
					if self.waveform=="sinus":
						self.alpha=self.amplitude*np.sin(2*np.pi*(t-t_step)*self.freq)+self.offset
					elif self.waveform=="triangle":
						self.alpha=(4*self.amplitude*self.freq)*((t-t_step)-(np.floor(2*self.freq*(t-t_step)+0.5))/(2*self.freq))*(-1)**(np.floor(2*self.freq*(t-t_step)+0.5))+self.offset
					elif self.waveform=="square":
						self.alpha=self.amplitude*np.sign(np.cos(2*np.pi*(t-t_step)*
									self.freq))+self.offset
					else:
						raise Exception("invalid waveform : use sinus,triangle or square")
					Array=pd.DataFrame([[t-self.t0,self.alpha]],columns=self.labels)
					try:
						for output in self.outputs:
							output.send(Array)
					except:
						pass
				self.step+=1
				if self.repeat and self.step==self.nb_step:
					self.step=0
				t_step=time.time()
  
class Streamer(MasterBlock):
	"""
Children class of MasterBlock. Send a fake stream of data in a pipe, with 
labels ["t(s)","signal"]
	"""
	
	
	def __init__(self,labels=['t(s)','signal']):
		"""
Send iterated value through a Link object.
		"""
		self.labels=labels
		
	def main(self):
		self.i=0
		while True:
			time.sleep(0.001)
			for output in self.outputs:
				output.send(pd.DataFrame([[time.time()-self.t0,self.i]],columns=self.labels))
			self.i+=1     

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


class StreamerComedi(MasterBlock):
	"""
Children class of MasterBlock. Send comedi value through a Link object.
	"""
	def __init__(self,comediSensor,labels=None,freq=8000,buffsize=10000):
		"""
This streamer read the value on all channels at the same time and send the 
values through a Link object. It can be very fast, but needs need an USB 2.0
port drove by ehci to work properly. xhci driver DOES NOT work (for now).

Parameters:
-----------
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
		self.comediSensor=comediSensor
		
		self.fd = self.c.comedi_fileno(self.comediSensor.device)	# get a file-descriptor

		self.BUFSZ = buffsize	# buffer size
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
		#if ret !=0: raise Exception("comedi_command failed...")

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
						Array=pd.DataFrame([array],columns=self.labels)
						for output in self.outputs:
							output.send(Array)

		except (KeyboardInterrupt):	
			self.comediSensor.close()


class VideoExtenso(MasterBlock): 
	"""
This class detects 4 spots, and evaluate the deformations Exx and Eyy.
	"""
	def __init__(self,camera="Ximea",white_spot=True,display=True,labels=['t(s)','Exx ()', 'Eyy()']):
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

		from skimage.segmentation import clear_border
		from skimage.morphology import label,erosion, square,dilation
		from skimage.measure import regionprops
		from skimage.filter import threshold_otsu, rank#, threshold_yen
		import cv2
		import SimpleITK as sitk
		self.cv2=cv2
		self.sitk=sitk
		go=False
		if camera=="Ximea":
			from crappy.technical import Ximea
			self.CameraClass=Ximea
		elif camera=="Jai":
			from crappy.technical import Jai
			self.CameraClass=Jai
		else:
			raise Exception("camera must be Ximea or Jai")
		###################################################################### camera INIT with ZOI selection
		self.white_spot=white_spot
		self.labels=labels
		self.display=display
		while go==False:
		# the following is to initialise the spot detection
			self.camera=self.CameraClass()
			self.xoffset=self.camera.sensor.xoffset
			self.yoffset=self.camera.sensor.yoffset
			image=self.camera.sensor.getImage()
			#print np.shape(image)
			self.camera.sensor.reset_ZOI()
			self.border=4 # Definition of the ZOI margin regarding the regionprops box			
			image=rank.median(image,square(15)) # median filter to smooth the image and avoid little reflection that may appear as spots.
			self.thresh = threshold_otsu(image) # calculate most effective threshold
			bw= image>self.thresh 
			#applying threshold
			if not (self.white_spot):
				bw=(1-bw).astype(np.uint8)
			#still smoothing
			bw = dilation(bw,square(3))
			bw = erosion(bw,square(3))
			# Remove artifacts connected to image border
			cleared = bw.copy()
			clear_border(cleared)
			# Label image regions
			label_image = label(cleared)
			borders = np.logical_xor(bw, cleared)
			label_image[borders] = -1
			# Create the empty vectors for corners of each ZOI
			regions=regionprops(label_image)
			#plt.imshow(bw,cmap=plt.cm.gray)
			#plt.show()
			print [region.area for region in regions]
			#mean_area=np.mean[region.area for region in regions]
			regions=[region for region in regions if region.area>100]
			self.NumOfReg=len(regions)
			if self.NumOfReg==4: 
				go=True
			else:	#	If detection goes wrong, start again
				print " Spots detected : ", self.NumOfReg
				self.camera.sensor.close()
				
		self.minx=np.empty([self.NumOfReg,1])
		self.miny=np.empty([self.NumOfReg,1])
		self.maxx=np.empty([self.NumOfReg,1])
		self.maxy=np.empty([self.NumOfReg,1])
		self.Points_coordinates=np.empty([self.NumOfReg,2])
		# Definition of the ZOI and initialisation of the regions border
		i=0
		for i,region in enumerate(regions): # skip small regions
			#if region.area > 100:
			self.minx[i], self.miny[i], self.maxx[i], self.maxy[i]= region.bbox	  	
		for i in range(0,self.NumOfReg): # find the center of every region
			self.Points_coordinates[i,0],self.Points_coordinates[i,1],self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]=self.barycenter_opencv(image[self.minx[i]-1:self.maxx[i]+1,self.miny[i]-1:self.maxy[i]+1],self.minx[i]-1,self.miny[i]-1)
		#	Evaluating initial distance bewteen 2 spots 
		self.L0x=self.Points_coordinates[:,0].max()-self.Points_coordinates[:,0].min()
		self.L0y=self.Points_coordinates[:,1].max()-self.Points_coordinates[:,1].min()
		
		
		#self.cv2.namedWindow('Press Esc to reconfigure the camera, anything else to start',self.cv2.WINDOW_NORMAL)
		#for i in range(0,self.NumOfReg): # For each region, plots the rectangle around the spot and a cross at the center
			#image = self.cv2.rectangle(image,(self.miny[i],self.minx[i]),(self.maxy[i],self.maxx[i]),(255,0,0),1)
			#image = self.cv2.circle(image,(int(self.Points_coordinates[i,1]),int(self.Points_coordinates[i,0])),1,(0,0,0),-1)
		#self.cv2.imshow('Press Esc to reconfigure the camera, anything else to start',image)
		#k=self.cv2.waitKey(0)
		#if k!=27: #	If the "esc" key is pressed
			#go=True
		#else:
			#self.camera.sensor.close()
		#self.cv2.destroyAllWindows()
		#self.cv2.destroyWindow('Press Esc to reconfigure the camera, anything else to start')
		#self.cv2.waitKey(1)
		
		#	evaluating global coordinate for next crop
		#self.cv2=reload(self.cv2)
		self.minx+=self.yoffset
		self.maxx+=self.yoffset
		self.miny+=self.xoffset
		self.maxy+=self.xoffset
		#image=self.camera.sensor.getImage()
		#	data for re-opening the camera device
		self.numdevice=self.camera.sensor.numdevice
		self.exposure=self.camera.sensor.exposure
		self.gain=self.camera.sensor.gain
		self.width=self.camera.sensor.width
		self.height=self.camera.sensor.height
		self.xoffset=self.camera.sensor.xoffset
		self.yoffset=self.camera.sensor.yoffset
		self.external_trigger=self.camera.sensor.external_trigger
		self.data_format=self.camera.sensor.data_format
		self.camera.sensor.close()
		


	def barycenter_opencv(self,image,minx,miny):
		"""
		computatition of the barycenter (moment 1 of image) on ZOI using OpenCV
		White_Mark must be True if spots are white on a dark material
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



class PID(MasterBlock):
	"""
	Work In Progress
	"""
	def __init__(self,actuators,P):
		self.actuators=actuators
		self.P=P
	def add_consigne(self,link):
		self.consigne=link
	def main(self):
		for input_ in self.inputs:
			Sensor=self.inputs[0].recv()
		t_init=time.time()-self.t0
		while True:
			Data=pd.DataFrame()
			for input_ in self.inputs:
				Data=pd.concat([Data,self.consigne.recv()])
				Sensor=self.inputs[0].recv()
				[Series.last_valid_index][2]
			

