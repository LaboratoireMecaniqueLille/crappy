from ._metaCondition import MetaCondition


class Filter(MetaCondition):
	def __init__(self,labels=[],mode="median",size=10):
		self.mode=mode
		self.size=size
		self.FIFO=[[] for label in labels]
		
	def evaluate(self,value):
		for i,label in enumerate(labels):
			FIFO[i].insert(0,value[label][0])
			if len(FIFO[i])>self.size:
				FIFO[i].pop()
			if method=="median":
				result=np.median(FIFO[i])
			elif method=="mean":
				result=np.mean(FIFO[i])
			value[label][0]=result
		return value

"""Receive a stream (multiprocessing.Value), filter it with said method and size, and return another Value (filtered_stream)
method : must be "median" or "mean"
size : number of values for floating mean or median
data_stream: input data as shared multiprocessing.Value
filtered_stream: output data as shared multiprocessing.Value
"""
