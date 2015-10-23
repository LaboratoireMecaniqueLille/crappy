from _meta import MasterBlock
#from multiprocessing import Process, Pipe
#import os
import numpy as np
import time
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import struct
#np.set_printoptions(threshold='nan', linewidth=500)
import pandas as pd
#import sys

class PID(MasterBlock):
	"""
	Work In Progress
	"""
	def __init__(self,actuators,P):
		self.actuators=actuators
		self.P=P
	def add_consigne(self,link):
		self.consigne=link
	def main(self):
		for input_ in self.inputs:
			Sensor=self.inputs[0].recv()
		t_init=time.time()-self.t0
		while True:
			Data=pd.DataFrame()
			for input_ in self.inputs:
				Data=pd.concat([Data,self.consigne.recv()])
				Sensor=self.inputs[0].recv()
				[Series.last_valid_index][2]
			

