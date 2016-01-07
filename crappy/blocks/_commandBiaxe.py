from _meta import MasterBlock

class CommandBiaxe(MasterBlock):
	"""Receive a signal and translate it for the Biaxe actuator"""
	def __init__(self, biaxe_technicals, speed=500):
		"""
Receive a signal and translate it for the Biaxe actuator.

CommandBiaxe(biaxe_technicals, speed)

Parameters
----------
biaxe_technicals : list of crappy.technical.Biaxe object.

speed: int, default = 500
		"""
		self.biaxe_technicals=biaxe_technicals
		self.speed=speed
		for biaxe_technical in self.biaxe_technicals:
			biaxe_technical.actuator.new()
	
	def main(self):
		try:
			last_cmd=0
			while True:
				Data=self.inputs[0].recv()
				#try:
					#cmd=Data['signal'].values[0]
				#except AttributeError:
				cmd=Data['signal']
				if cmd!= last_cmd:
					for biaxe_technical in self.biaxe_technicals:
						biaxe_technical.actuator.set_speed(cmd*self.speed)
					last_cmd=cmd
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in measureComediByStep : ", e
			for biaxe_technical in self.biaxe_technicals:
				biaxe_technical.actuator.set_speed(0)
				biaxe_technical.actuator.close_port()
			raise