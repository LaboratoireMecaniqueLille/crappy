# coding: utf-8
from _meta import MasterBlock
import time
import os
#import gc

class CommandComedi(MasterBlock):
	"""Receive a signal and translate it for the Biotens actuator"""
	def __init__(self, comedi_actuators):
		"""
Receive a signal and translate it for the Comedi card.

CommandComedi(comedi_actuators)

Parameters
----------
comedi_actuators : list of crappy.actuators.ComediActuator objects.
		"""
		self.comedi_actuators=comedi_actuators
		print "command comedi! "
	
	def main(self):
		print "command comedi! :", os.getpid()
		try:
			last_cmd=0
			while True:
				Data=self.inputs[0].recv()
				#try:
					#cmd=Data['signal'].values[0]
				#except AttributeError:
				cmd=Data['signal']
				if cmd!= last_cmd:
					for comedi_actuator in self.comedi_actuators:
						comedi_actuator.set_cmd(cmd)
					last_cmd=cmd
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandComedi : ", e
			for comedi_actuator in self.comedi_actuators:
				comedi_actuator.close()
			#raise
