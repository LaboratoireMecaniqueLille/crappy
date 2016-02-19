# coding: utf-8
from multiprocessing import Process, Pipe
#import os
import ctypes, time
libc = ctypes.CDLL('libc.so.6')

class MasterBlock(object):
	"""
	Main class for block architecture. All blocks should inherit this class.	

	Methods
	-------
	main()
		Override it to define the main function of this block.
	add_input(Link object)
		Add a Link object as input.
	add_output(Link object)
		Add a Link as output.
	start()
		Start the main() method as a Process.
	stop()
		Stops the process.
	"""
	instances = []
	def __new__(cls, *args, **kwargs): #	Keeps track of all instances
		instance = super(MasterBlock, cls).__new__(cls, *args, **kwargs)
		instance.instances.append(instance)
		return instance
	
	def main(self):
		pass
	
	def add_output(self,link):
		try: # test if the outputs list exist
			a_=self.outputs[0]
		except AttributeError: # if it doesn't exist, create it
			self.outputs=[]
		self.outputs.append(link)
			
	def add_input(self,link):
		try: # test if the outputs list exist
			a_=self.inputs[0]
		except AttributeError: # if it doesn't exist, create it
			self.inputs=[]
		self.inputs.append(link)
	
	def start(self):
		try:
			self.proc=Process(target=self.main,args=())
			self.proc.start()
		except (Exception,KeyboardInterrupt) as e:
			print "Exception : ", e
			self.proc.terminate()
			#raise #raise the error to the next level for global shutdown
		
	#def join(self):
		#self.proc.join()
	def stop(self):
		self.proc.terminate()
		
	def set_t0(self,t0):
		self.t0=t0


def delay(ms):
  """Delay in milliseconds with libc usleep() using ctypes.
  It has a better repeatability than time.sleep()"""
  ms = int(ms*1000)
  libc.usleep(ms)