import abc	

class MetaCondition:
	"""metaclass for all Links conditions. Must implement the evaluate method"""
	__metaclass__= abc.ABCMeta

	@abc.abstractmethod
	def evaluate(self):
		pass