# coding: utf-8
from _meta import MasterBlock
import time
import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError

class Streamer(MasterBlock):
	"""
Send a fake stream of data.
	"""
	def __init__(self,labels=['t(s)','signal']):
		"""
Use it for testing.

Parameters
----------
labels : list of str, default = ['t(s)','signal']
	Output labels.
		"""
		self.labels=labels
		
	def main(self):
		self.i=0
		while True:
			time.sleep(0.001)
			try:
				for output in self.outputs:
					output.send(OrderedDict(zip(self.labels,[time.time()-self.t0,self.i])))
			except TimeoutError:
				raise
			except AttributeError: #if no outputs
				pass
			self.i+=1     
