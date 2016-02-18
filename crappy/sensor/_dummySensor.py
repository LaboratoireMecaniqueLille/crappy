# coding: utf-8
import numpy as np
import time
#import comedi as c
from multiprocessing import Array
#import os
import sys, string, struct



class DummySensor(object):
	"""
	Sensor class for Comedi devices.
	"""
	def __init__(self,*args,**kwargs):
		self.args=args
		self.kwargs=kwargs
		
	 
	def new(self,*args):
		pass

	def getData(self,*args):
		t=time.time()
		return t
			
	def close(self,*args):
		pass
	
