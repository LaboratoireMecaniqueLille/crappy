from _meta import MasterBlock
import pandas as pd
import os
#import gc
from collections import OrderedDict

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
		try:
			print "compacter!", os.getpid()
			while True:
				#data=[0 for x in xrange(self.acquisition_step)]
				for i in range(self.acquisition_step):
					if i==0:
						Data=self.inputs[0].recv()
					else:
						Data1=self.inputs[0].recv()
					if len(self.inputs)!=1:
						for k in range(1,len(self.inputs)):
							data_recv=self.inputs[k].recv()
							#try:
								#if i ==0:
									#Data=pd.concat([Data,data_recv],axis=1)
								#else:
									#Data1=pd.concat([Data1,data_recv],axis=1)
							#except AttributeError:
							if i ==0:
								Data.update(data_recv)
							else:
								Data1.update(data_recv)
					if i!=0:
						try:
							Data=OrderedDict(zip(Data.keys(),[Data.values()[t]+(Data1.values()[t],) for t in range(len(Data.keys()))]))
						except TypeError:
							Data=OrderedDict(zip(Data.keys(),[(Data.values()[t],)+(Data1.values()[t],) for t in range(len(Data.keys()))]))
				for j in range(len(self.outputs)):
					self.outputs[j].send(Data)
				#gc.collect()
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in Compacter %s: %s" %(os.getpid(),e)
			raise