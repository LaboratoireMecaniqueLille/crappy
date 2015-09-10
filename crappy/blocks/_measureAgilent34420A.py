from _meta import MasterBlock
import time
import pandas as pd

class MeasureAgilent34420A(MasterBlock):
	"""
Children class of MasterBlock. Send value through a Link object.
	"""
	def __init__(self,agilentSensor,labels=['t(s)','R'],freq=None):
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

		except Exception as e:
			print "Exception in measureAgilent34420A : ", e
			self.agilentSensor.close()
			raise

