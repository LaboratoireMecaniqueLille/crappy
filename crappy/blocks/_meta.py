from multiprocessing import Process, Pipe

class MasterBlock(object):
	"""
Main class for block architecture. All blocks should inherit this class.
	
Methods:
--------
main() : override it to define the main function of this block.
add_input(Link object): add a Link object as input.
add_output(Link object) : add a Link as output.
start() : start the main() method as a Process. It is designed to catch all 
	exceptions/error and terminate the process for safety. It also 	re-raises 
	the exception/error for upper-level handling.
stop() : stops the process.
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
			raise #raise the error to the next level for global shutdown
		
	def join(self):
		self.proc.join()
		
	def stop(self):
		self.proc.terminate()
		
	def set_t0(self,t0):
		self.t0=t0