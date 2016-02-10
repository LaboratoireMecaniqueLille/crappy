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
	def __init__(self,P,I,D,ControllerDirection,outMin=-10,outMax=10):
		
		self.kp=P
		self.ki=I
		self.kd=D
		self.inAuto=False
		self.t_0=time.time()
		self.outMin=outMin
		self.outMax=outMax
		#self.Iterm=0
		#self.mode='Off'
		#self.lastOutput=0
	#def add_setpoint(self,link):
		#self.setpoint=link
	
	##def setMode(mode):
		##newMode=self.mode
		##if newMode is 'On' or newMode is 'Off':
			##if newMode is not self.mode and newMode is 'On'
				##self.initialize()
				##self.inAuto=True
			##else if newMode is 'Off':
				##self.inAuto=False
	
	#def initialize():
		#self.Iterm=self.lastOutput

		#if self.Iterm > outMax:
			#self.Iterm = outMax
		#else if self.Iterm < outMin:
			#self.Iterm=outMin
	
	#def compute()
		#if self.inAuto is True:
			#now=time.time()
			#timeChange=now-self.lastTime
			#self.input_=self.inputs[0].recv()
			#self.error=self.setpoint-self.input_
			#self.ki*=timeChange/self.lastTimeChange
			#self.kd/=timeChange/self.lastTimeChange
			#self.Iterm = self.ki * error*timeChange
			#if self.Iterm > outMax:
				#self.Iterm = outMax
			#else if self.Iterm < outMin:
				#self.Iterm=outMin
			#dInput=self.input_-self.lastInput
			#self.output=self.kp * error + self.Iterm - self.kd * dInput/timeChange
			
			#if self.output > outMax:
				#self.output = outMax
			#else if self.output < outMin:
				#self.output = outMin
			#self.lastOutput=self.output
			#Array=pd.DataFrame([[now-self.t_0, self.output]])
			#try:
				#for output in self.outputs:
					#output.send(Array)
			#except:
				#pass
			#self.lastInput = self.input_
			#self.lastTime = now
			#self.lastTimeChange=timeChange
			#return True
		#else return False
	

		
	
	#def main(self):
		#for input_ in self.inputs:
			#Sensor=self.inputs[0].recv()
		#t_init=time.time()-self.t0
		#self.lastTime=time.time()
		#while True:
			#self.compute()
			##Data=pd.DataFrame()
			##for input_ in self.inputs:
				##Data=pd.concat([Data,self.consigne.recv()])
				##Sensor=self.inputs[0].recv()
				##[Series.last_valid_index][2]
			

