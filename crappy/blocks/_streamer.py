from _meta import MasterBlock
import time
import pandas as pd
from collections import OrderedDict

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
				output.send(OrderedDict(zip(self.labels,[time.time()-self.t0,self.i])))
			self.i+=1     
