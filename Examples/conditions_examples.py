class condition_cycle(modules.links.Condition):
	def __init__(self):
		self.cycle=0
		self.go=True
		
	def evaluate(self,value):
		if value[2]>=F_max and self.go==True:
			self.cycle+=1
			self.go=False
		if value[2]<=F_min and self.go==False:
			self.go=True
		return self.cycle

class condition_cycle_bool(modules.links.Condition):
	def __init__(self,n=1):
		self.cycle=0
		self.go=True
		self.n=n
		
	def evaluate(self,value):
		if value[2]>=F_max and self.go==True:
			self.cycle+=1
			self.go=False
			if self.cycle%self.n==0:
				return True
			else:
				return False
		elif value[2]<=F_min and self.go==False:
			self.go=True
			if self.cycle%self.n==0:
				return True
			else:
				return False
		else:
			return False


class condition_camera(modules.links.Condition):
	def __init__(self,n=1):
		self.cycle=0
		self.n=n
		
	def evaluate(self,value):
		if self.cycle%self.n==0:
			#print self.cycle
			self.cycle+=1
			return value 
		else:
			#print self.cycle
			self.cycle+=1
			return None
		#self.cycle+=1