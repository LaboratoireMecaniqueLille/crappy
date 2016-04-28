# coding: utf-8
from ..sensor import _ximeaSensor
#from ..actuator import ximeaActuator
#import numpy as np
#import cv2
#import time

class Ximea(object):
	"""
DEPRECATED : use technicalCamera instead.
	"""
	def __init__(self, numdevice=0, exposure=10000, gain=0, width=2048, height=2048, xoffset=0, yoffset=0, framespersec=None, external_trigger=False, data_format=0):
		self.exposure= exposure
		self.gain=gain
		self.FPS=framespersec
		self.framespersec=self.FPS
		self.numdevice = numdevice
		self.width=width
		self.height=height
		self.xoffset=xoffset
		self.yoffset=yoffset
		self.external_trigger=external_trigger
		self.data_format=data_format
		self.sensor=_ximeaSensor.XimeaSensor(self.numdevice, self.exposure, self.gain, self.width, self.height, self.xoffset, self.yoffset, self.framespersec, self.external_trigger, self.data_format)
		self.actuator=None