# coding: utf-8
from multiprocessing import Pipe
import copy
from functools import wraps
#import errno
#import os
import signal

class TimeoutError(Exception):
	"""Custom error to raise in case of timeout."""
	pass

def timeout_func(f):
	"""Decorator for adding a timeout to a link send."""
	def _handle_timeout(signum, frame):
		raise TimeoutError("timeout error in pipe send")
	
	def wrapper(*args):
		signal.signal(signal.SIGALRM, _handle_timeout)
		signal.setitimer(signal.ITIMER_REAL,args[0].timeout) # args[0] is the class "self" here.
		try:
			result = f(*args)
		finally:
			signal.alarm(0)
		return result
	return wrapper


class Link(object):
	"""
Link class. All connection between Blocks should be made with this.
	"""
	def __init__(self,condition=None,timeout=0.1,action="warn"):
		"""
Creates a pipe and is used to transfer information between Blocks using a pipe.
You can add one or multiple conditions to modify the value transfered.


Parameters
----------
condition : Children class of links.MetaCondition, optionnal
	Each "send" call will pass through the condition.evaluate method and sends
	the returned value.
	You can pass a list of conditions, the link will execute them in order.
	
timeout : int or float, default = 0.1
	Timeout for the send method.
	
action : {'warn','kill'}, default = "warn"
	Action in case of TimeoutError in the send method. You can warn only or 
	choose to kill the link.


Attributes
----------
in_ : 
	Input extremity of the pipe.
	
out_ : 
	Output extremity of the pipe.
	
external_trigger : Default=None
	Can be add through "add_external_trigger" instance

		"""
		
		self.in_,self.out_=Pipe(duplex=False)
		self.external_trigger=None
		self.condition=condition
		self.timeout=timeout
		self.action=action
	
	def add_external_trigger(self,link_instance):
		"""Add an external trigger Link."""
		self.external_trigger=link_instance
		self.condition.external_trigger=link_instance
	
	def send(self,value):
		"""Send the value, or a modified value if you pass it through a 
		condition."""
		try:
			self.send_timeout(value)
		except TimeoutError as e:
			if self.action=="warn":
				print "WARNING : Timeout error in pipe send!"
			elif self.action=="kill":
				print "Killing Link : ", e
				raise
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise
			
			
	@timeout_func
	def send_timeout(self,value):
		try:
			if self.condition==None:
				self.out_.send(value)
			else:
				try:
					for i in range(len(self.condition)):
						value=self.condition[i].evaluate(copy.copy(value))
				except TypeError: # if only one condition
					value=self.condition.evaluate(copy.copy(value))
				if not value is None:
					self.out_.send(value)
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