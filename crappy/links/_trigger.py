from ._metaCondition import MetaCondition


class Trigger(MetaCondition):
	def __init__(self,label=[],output=None):
		self.label=label
		self.output=output
		self.first=True
		#self.FIFO=[[] for label in labels]
		
	def evaluate(self,value):
		if self.first: # init values
			self.val=value[self.label][0]
			self.last_val=self.val
			self.first=False
		self.val=value[self.label][0]
		if self.val!=self.last_val:
			self.last_val=self.val
			if self.output is not None:
				return self.output
			else:
				return value
		else:
			return None