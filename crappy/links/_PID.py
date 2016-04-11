# coding: utf-8
from ._metaCondition import MetaCondition
import time


class PID(MetaCondition):
	"""WIP, not working yet."""
	def __init__(self,P,I,D,label_consigne,label_retour,outMin=-10,outMax=10):
		self.P=P
		self.I=I
		self.D=D
		self.label_consigne=label_consigne
		self.label_retour=label_retour
		self.outMin=outMin
		self.outMax=outMax
		self.first=True

		
	def evaluate(self,value):
		#self.t=time.time()
		self.retour=self.external_trigger.recv()[self.label_retour]
		self.consigne=value.pop(self.label_consigne)
		#print self.retour,self.consigne
		if self.first:
			self.lastTime=time.time()
			self.last_retour=self.retour
			self.first=False
			self.lastTimeChange=10**118 # for initialization
		self.compute()
		val=self.output+self.retour
		if val > self.outMax:
			val = self.outMax
		elif val < self.outMin:
			val = self.outMin
		value[self.label_consigne]=val
		#print value
		return value
	
	def initialize(self):
		self.Iterm=self.lastOutput

		if self.Iterm > outMax:
			self.Iterm = outMax
		elif self.Iterm < outMin:
			self.Iterm=outMin
	
	def compute(self):
		#if self.inAuto is True:
		now=time.time()
		timeChange=now-self.lastTime
		#self.input_=self.inputs[0].recv()
		self.error=self.consigne-self.retour
		self.I*=timeChange/self.lastTimeChange
		self.D/=timeChange/self.lastTimeChange
		self.Iterm = self.I * self.error*timeChange
		if self.Iterm > self.outMax:
			self.Iterm = self.outMax
		elif self.Iterm < self.outMin:
			self.Iterm=self.outMin
		dInput=self.retour-self.last_retour
		self.output=self.P*self.error+self.Iterm+self.D*dInput/timeChange
		if self.output > self.outMax:
			self.output = self.outMax
		elif self.output < self.outMin:
			self.output = self.outMin
		self.lastOutput=self.output
		self.last_retour = self.retour
		self.lastTime = now
		self.lastTimeChange=timeChange

			
	##def setMode(mode):
		##newMode=self.mode
		##if newMode is 'On' or newMode is 'Off':
			##if newMode is not self.mode and newMode is 'On'
				##self.initialize()
				##self.inAuto=True
			##else if newMode is 'Off':
				##self.inAuto=False

		
	
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
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		#value[self.label_consigne]=val*self.coeff
		#return value