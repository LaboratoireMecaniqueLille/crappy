# coding: utf-8
#import numpy as np
import time
#import comedi as c
#from multiprocessing import Array
#import os
#import sys, string, struct



class DummySensor(object):
	"""Mock a sensor and return the time. Use it for testing."""
	def __init__(self,*args,**kwargs):
		self.args=args
		self.kwargs=kwargs
		
	def new(self,*args):
		"""Do nothing."""
		pass

	def getData(self,*args):
		"""Return time."""
		t=time.time()
		return t
			
	def close(self,*args):
		"""Do nothing."""
		pass
	
