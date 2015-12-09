from ._metaCondition import MetaCondition
import numpy as np

class Filter(MetaCondition):
	def __init__(self,labels=[],mode="median",size=10):
		self.mode=mode
		self.size=size
		self.labels=labels
		self.FIFO=[[] for label in self.labels]
		#print self.FIFO
		
	def evaluate(self,value):
		for i,label in enumerate(self.labels):
			#print self.FIFO[i]
			self.FIFO[i].insert(0,value[label])
			if len(self.FIFO[i])>self.size:
				self.FIFO[i].pop()
			if self.mode=="median":
				result=np.median(self.FIFO[i])
			elif self.mode=="mean":
				result=np.mean(self.FIFO[i])
			value[label+"_filtered"]=result
		return value

"""Receive a stream (multiprocessing.Value), filter it with said method and size, and return another Value (filtered_stream)
method : must be "median" or "mean"
size : number of values for floating mean or median
data_stream: input data as shared multiprocessing.Value
filtered_stream: output data as shared multiprocessing.Value
"""
