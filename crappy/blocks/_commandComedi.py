from _meta import MasterBlock

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
	
	def main(self):
		try:
			last_cmd=0
			while True:
				Data=self.inputs[0].recv()
				cmd=Data['signal'].values[0]
				if cmd!= last_cmd:
					for comedi_actuator in self.comedi_actuators:
						comedi_actuator.set_cmd(cmd)
					last_cmd=cmd
		except Exception as e:
			print "Exception in CommandComedi : ", e
			for comedi_actuator in self.comedi_actuators:
				comedi_actuator.close()
			raise
