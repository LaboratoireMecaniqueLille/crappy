# coding: utf-8
from ._metaCondition import MetaCondition

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
		
	def evaluate(self,value):
		self.t=time.time()
		retour=self.external_trigger.recv()[self.label_retour]
		consigne=value.pop(self.label_consigne)
		t_init=time.time()-self.t0
		self.lastTime=time.time()
		while True:
			self.compute()
		
	##def setMode(mode):
		##newMode=self.mode
		##if newMode is 'On' or newMode is 'Off':
			##if newMode is not self.mode and newMode is 'On'
				##self.initialize()
				##self.inAuto=True
			##else if newMode is 'Off':
				##self.inAuto=False
	
	def initialize(self):
		self.Iterm=self.lastOutput

		if self.Iterm > outMax:
			self.Iterm = outMax
		elif self.Iterm < outMin:
			self.Iterm=outMin
	
	def compute(self):
		if self.inAuto is True:
			now=time.time()
			timeChange=now-self.lastTime
			self.input_=self.inputs[0].recv()
			self.error=self.setpoint-self.input_
			self.ki*=timeChange/self.lastTimeChange
			self.kd/=timeChange/self.lastTimeChange
			self.Iterm = self.ki * error*timeChange
			if self.Iterm > outMax:
				self.Iterm = outMax
			elif self.Iterm < outMin:
				self.Iterm=outMin
			dInput=self.input_-self.lastInput
			self.output=self.kp * error + self.Iterm - self.kd * dInput/timeChange
			
			if self.output > outMax:
				self.output = outMax
			elif self.output < outMin:
				self.output = outMin
			self.lastOutput=self.output
			Array=pd.DataFrame([[now-self.t_0, self.output]])
			try:
				for output in self.outputs:
					output.send(Array)
			except:
				pass
			self.lastInput = self.input_
			self.lastTime = now
			self.lastTimeChange=timeChange
			return True
		else:
			return False
	

		
	
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