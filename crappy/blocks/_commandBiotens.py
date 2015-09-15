from _meta import MasterBlock
import time

class CommandBiotens(MasterBlock):
	"""Receive a signal and translate it for the Biotens actuator"""
	def __init__(self, biotens_technicals, speed=5):
		"""
Receive a signal and translate it for the Biotens actuator.

CommandBiotens(biotens_technical,speed=5)

Parameters
----------
biotens_technicals : list of crappy.technical.Biotens object.

speed: int
		"""
		self.biotens_technicals=biotens_technicals
		self.speed=speed
		for biotens_technical in self.biotens_technicals:
			biotens_technical.actuator.clear_errors()
	
	def main(self):
		try:
			last_cmd=0
			while True:
				Data=self.inputs[0].recv()
				cmd=Data['signal'].values[0]
				if cmd!= last_cmd:
					for biotens_technical in self.biotens_technicals:
						biotens_technical.actuator.setmode_speed(cmd*self.speed)
					last_cmd=cmd
				for biotens_technical in self.biotens_technicals:
					position=biotens_technical.sensor.read_position()
					Array=pd.DataFrame([[time.time()-self.t0,position]],columns=['t(s)','position'])
					try:
						for output in self.outputs:
							output.send(Array)
					except:
						pass
				
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandBiotens : ", e
			for biotens_technical in self.biotens_technicals:
				biotens_technical.actuator.stop_motor()
			raise