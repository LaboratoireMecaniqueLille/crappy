from _meta import MasterBlock
import time
#import pandas as pd
from collections import OrderedDict

class CommandBiotens(MasterBlock):
	"""Receive a signal and translate it for the Biotens actuator"""
	def __init__(self, biotens_technicals, speed=5):
		"""
Receive a signal and translate it for the Biotens actuator.

CommandBiotens(biotens_technical,speed=5)

Parameters
----------
biotens_technicals : list of crappy.technical.Biotens object.

speed: int, default = 5
		"""
		self.biotens_technicals=biotens_technicals
		self.speed=speed
		for biotens_technical in self.biotens_technicals:
			biotens_technical.actuator.clear_errors()
	
	def main(self):
		try:
			#print "top command"
			last_cmd=0
			self.last_time=self.t0
			while True:
#<<<<<<< HEAD
				Data=self.inputs[0].recv()
				#try:
					#cmd=Data['signal'].values[0]
				#except AttributeError:
				cmd=Data['signal']
				if cmd!= last_cmd:
					for biotens_technical in self.biotens_technicals:
						biotens_technical.actuator.setmode_speed(cmd*self.speed)
					last_cmd=cmd
#=======
				##print "top command2"
				#Data=self.inputs[0].recv(blocking=False)
				#try :
					#cmd=Data['signal'].values[0]
					##print "cmd : ", cmd
					#if cmd!= last_cmd:
						#for biotens_technical in self.biotens_technicals:
							#biotens_technical.actuator.setmode_speed(cmd*self.speed)
						#last_cmd=cmd
				#except TypeError:
					#pass
#>>>>>>> da574d1a068d65b39ade18b0a76722785c27f855
				t=time.time()
				if (t-self.last_time)>=0.2:
					#print "top command3"
					self.last_time=t
					for biotens_technical in self.biotens_technicals:
						position=biotens_technical.sensor.read_position()
					#Array=pd.DataFrame([[t-self.t0,position]],columns=['t(s)','position'])
					Array=OrderedDict(zip(['t(s)','position'],[t-self.t0,position]))
					try:
						for output in self.outputs:
							#print "sending position ..."
							output.send(Array)
							#print "position sent ..."
					except:
						#print "no outputs"
						pass
				
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandBiotens : ", e
			for biotens_technical in self.biotens_technicals:
				biotens_technical.actuator.stop_motor()
			raise