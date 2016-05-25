# coding: utf-8
from _meta import MasterBlock
import time
#import pandas as pd
from collections import OrderedDict
from ..links._link import TimeoutError

class CommandBiotens(MasterBlock):
	"""Receive a signal and send it for the Biotens actuator"""
	def __init__(self, biotens_technicals, signal_label='signal', speed=5):
		"""
Receive a signal and translate it for the Biotens actuator.

Parameters
----------
biotens_technicals : list of crappy.technical.Biotens objects.
	List of all the axes to control.
signal_label : str, default = 'signal'
	Label of the data to be transfered.
speed: int, default = 5
	Wanted speed, in mm/min.
		"""
		self.biotens_technicals=biotens_technicals
		self.speed=speed
		self.signal_label=signal_label
		for biotens_technical in self.biotens_technicals:
			biotens_technical.clear_errors()
	
	def main(self):
		try:
			#print "top command"
			last_cmd=0
			self.last_time=self.t0
			while True:
				Data=self.inputs[0].recv()
				#try:
					#cmd=Data['signal'].values[0]
				#except AttributeError:
				cmd=Data[self.signal_label]
				if cmd!= last_cmd:
					for biotens_technical in self.biotens_technicals:
						biotens_technical.actuator.set_speed(cmd*self.speed)
					last_cmd=cmd
				t=time.time()
				if (t-self.last_time)>=0.2:
					#print "top command3"
					self.last_time=t
					for biotens_technical in self.biotens_technicals:
						position=biotens_technical.sensor.get_position()
					#Array=pd.DataFrame([[t-self.t0,position]],columns=['t(s)','position'])
					Array=OrderedDict(zip(['t(s)','position'],[t-self.t0,position]))
					try:
						for output in self.outputs:
							#print "sending position ..."
							output.send(Array)
					except TimeoutError:
						raise
					except AttributeError: #if no outputs
						pass
				
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandBiotens : ", e
			for biotens_technical in self.biotens_technicals:
				biotens_technical.actuator.stop_motor()
			#raise