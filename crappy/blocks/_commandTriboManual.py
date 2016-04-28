from _meta import MasterBlock

class CommandTriboMaintien(MasterBlock):
	def __init__(self, VariateurTribo, comediDigital):
		self.VariateurTribo=VariateurTribo
		self.comediDigital=comediDigital
	def main(self):
		t_0=time.time()
		print "t0 = ", t_0
		datastring=''
		self.VariateurTribo.init=False
		consigne=0.0
		self.VariateurTribo.go_effort(consigne)
		while 1:
			print 'Enter a command :'
			input_=stdin.readline()
			if input_== 'init\n':
				self.VariateurTribo.initialisation()
	
			elif input_== 'effort\n' and self.VariateurTribo.mode!='effort' and self.VariateurTribo.init:
				self.VariateurTribo.set_mode_analog()
				self.comediDigital.On()
			
			elif input_=='position' and self.VariateurTribo.mode!='effort' and self.VariateurTribo.init:
				self.VarœiateurTribo.set_mode_position()
				self.comediDigital.Off()
				self.VariateurTribo.read_position()
				
			elif input_='stop':
				self.VariateurTribo.stop_motor()
			elif self.VariateurTribo.mode=='effort' and self.VariateurTribo.init:
				try:
					consigne=long(input_)
					t_1=time.time()
					print 't_1 =',t_1-t_0
					if(consigne<=5000 and consigne>=0):
						consigne=consigne*10
						print consigne
						self.VariateurTribo.go_effort(consigne)
					else:
						print 'Consigne set too high or too low'
				except:
					print 'Pls enter a valid number'
			elif self.VariateurTribo.mode=='position' and self.VariateurTribo.init:
				try:
					consigne=int(input_)
					print consigne
					if(consigne<=6500 and consigne>=-20000):
						self.VariateurTribo.go_position(consigne)
					else:
						print "pls enter a number between 6500 and -20000"
				except:
					print "enter a valid number"
			elif self.VariateurTribo.init:
				print "pls enter a valid command"
			else:
				print "pls initialise"


newstdin=os.fdopen(os.dup(sys.stdin.fileno()))