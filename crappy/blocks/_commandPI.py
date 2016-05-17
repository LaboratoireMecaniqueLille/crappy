# coding: utf-8
from _meta import MasterBlock

class CommandPI(MasterBlock):
	"""Receive a signal and send it for the Biaxe actuator"""
	def __init__(self, PI_actuators, signal_label='signal',initial_command=0):
		"""
Receive a signal and translate it for the Biaxe actuator.

Parameters
----------
biaxe_technicals : list of crappy.technical.Biaxe objects
	List of all the axes to control.
signal_label : str, default = 'signal'
	Label of the data to be transfered.
speed: int, default = 500
	Wanted speed. 1 is equivalent to a speed of 0.002 mm/s.
		"""
		self.PI_actuators=PI_actuators
		self.signal_label=signal_label
		self.initial_command=initial_command

	
	def main(self):
		try:
			last_cmd=self.initial_command
			while True:
				Data=self.inputs[0].recv()
				#try:
					#cmd=Data['signal'].values[0]
				#except AttributeError:
				cmd=Data[self.signal_label]
				if cmd!= last_cmd:
					self.PI_actuators.set_absolute_disp(cmd)
					last_cmd=cmd
				self.PI_actuators.ser.write("%c%cTP\r"%(1,'0')) #connaitre la position
				#print "position platine : ", self.PI_actuators.ser.readline()
				#print "commande platine : ", cmd
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandPI : ", e
			self.PI_actuators.close_port()
			#raise