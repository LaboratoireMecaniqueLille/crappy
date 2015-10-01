from multiprocessing import Pipe

class Link(object):
	"""
Main class for links. All links should inherit this class.
	"""
	
	
	def __init__(self,condition=None):
		"""
Link([condition=None])

Creates a pipe with a condition as attribute, and is used to transfer 
information between blocks using a pipe, triggered by the condition.

Parameters
----------
condition : Children class of links.Condition, optionnal
	Each "send" call will pass through the condition.evaluate method and sends
	the returned value.
	
Attributes
----------
in_ : input extremity of the pipe.
out_ : output extremity of the pipe.
external_trigger : Default=None, can be add through "add_external_trigger" instance

Methods
-------
add_external_trigger(link_instance): add an external trigger Link.
send(pickable) : sends a pickable object (or the boolean returned by the 
	condition).
recv(blocking=True) : receive a pickable object. If blocking=False, return None
if there is no data
		"""
		
		self.in_,self.out_=Pipe(duplex=True)
		self.external_trigger=None
		self.condition=condition
	
	def add_external_trigger(self,link_instance):
		self.external_trigger=link_instance
		self.condition.external_trigger=link_instance
	
	def send(self,value):
		"""Send data through the condition.evaluate(value) function"""
		try:
			if self.condition==None:
				self.out_.send(value)
			else:
				#if self.external_trigger==None:
				val=self.condition.evaluate(value)
				if not val is None:
					self.out_.send(val)
				#else:
					#val=self.condition.evaluate(value,self.external_trigger)
					#if val is not None:
						#self.out_.send(val)
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise
	
	def recv(self,blocking=True):
		"""Receive data. If blocking=False, return None if there is no data"""
		try:
			if blocking:
				return self.in_.recv()
			else:
				if self.in_.poll():
					return self.in_.recv()
				else:
				  return None
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise