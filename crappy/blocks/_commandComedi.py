from _meta import MasterBlock
import time
import os

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
			#t_max=0
			#t_mean=0
			#k=1
			last_cmd=0
			i=1
			loop_mean=0
			loop_max=0
			last_t=self.t0
			while True:
				#t_1=time.time()
				Data=self.inputs[0].recv()
				#t_recv=time.time()-t_1
				#t_max=max(t_max,t_recv)
				#t_mean+=t_recv
				#if k%500==0:
					#print "t_max, t_mean tension: ", t_max,t_mean/k
					#t_max=0
				#k+=1
				cmd=Data['signal'].values[0]
				#t_ori=Data['t(s)'][0]
				#t_now=time.time()-self.t0
				#delta+=(t_now-t_ori)
				#
					#print "delta comedi = ", (delta/i)
				if cmd!= last_cmd:
					for comedi_actuator in self.comedi_actuators:
						comedi_actuator.set_cmd(cmd)
					last_cmd=cmd
				t_loop=time.time()
				loop_mean+=t_loop-last_t
				loop_max=max(loop_max,t_loop-last_t)
				if i%500==0:
					#print "t_loop sending comedi mean, max:                                                ", loop_mean/i,loop_max
					loop_max=0
				last_t=t_loop
				i+=1
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in CommandComedi : ", e
			for comedi_actuator in self.comedi_actuators:
				comedi_actuator.close()
			raise
