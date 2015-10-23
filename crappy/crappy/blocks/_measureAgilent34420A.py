from _meta import MasterBlock
import time
import pandas as pd
import os

class MeasureAgilent34420A(MasterBlock):
	"""
Children class of MasterBlock. Send value through a Link object.
	"""
	def __init__(self,agilentSensor,labels=['t_agilent(s)','R'],freq=None):
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
			#t_max=0
			#t_mean=0
			#k=1
			print "mesureagilent " , os.getpid()
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
					Data=pd.DataFrame([data],columns=self.labels)
				if trigger=="external":
					#t_1=time.time()
					Data = self.inputs[0].recv() # wait for a signal
					#t_recv=time.time()-t_1
					#t_max=max(t_max,t_recv)
					#t_mean+=t_recv
					#if k%10==0:
						#print "t_max, t_mean tension: ", t_max,t_mean/k
						#t_max=0
					#k+=1
					if Data is not None:
						#print "top res3"
						ret=self.agilentSensor.getData()
						if ret == False:
							ret=np.nan
						Data[self.labels[0]] = pd.Series((time.time()-self.t0), index=Data.index) # verify if timestamps really differ and delete this line
						Data[self.labels[1]] = pd.Series((ret), index=Data.index) # add one column
				#Array=pd.DataFrame([data],columns=self.labels)
				#Data.append(Array)
				if trigger=="internal" or Data is not None:
					#print "top res4"
					for output in self.outputs:
						output.send(Data)

		except (Exception,KeyboardInterrupt) as e:
			print "Exception in measureAgilent34420A : ", e
			self.agilentSensor.close()
			raise

