# # coding: utf-8
# from sys import stdin
#
# import time
#
# from _meta import MasterBlock
#
# class CommandTriboMaintien(MasterBlock):
# 	def __init__(self, VariateurTribo, comediOut):
# 		self.VariateurTribo=VariateurTribo
#
#
# 	def main(self):
#
# 		#paramètres essai
# 		cycles=1
# 		Vmax=1000.0
# 		Vmin=0.0
# 		F=500.0
# 		inertia=2.0
#
#
#
# 		t_0=time.time()
# 		print "t0 = ", t_0
# 		self.VariateurTribo.init=False
#
# 		while 1:
# 			print 'Enter a command :'
# 			input_=stdin.readline()
# 			if input_== 'init\n':
# 				self.VariateurTribo.initialisation()
# 				print 'The tribometer has been initialised'
#
# 			elif self.VariateurTribo.init:
# 				print "Do you want to begin the test? (Y/N)"
# 				input_= stdin.readline()
# 				if input_ == "Y":
# 					cycle=1
# 					while cycle<=cycles:
# 						print 'Cycle n°'+cycle
# 						print 'Acceleration'
# 						comediOut.set_cmd(Vmax)
#
# 						print 'Wait for speed stabilisation'
# 						while self.input[0].recv()!=Vmax:
# 							pass
# 						time.sleep(2)
# 						print 'Braking'
# 						t_1=time.time()
# 						while self.input[0].recv()>Vmin:
# 							comediOut.set_cmd(self.input[0].recv()-self.input[1].recv()*(t_1-time.time())/inertia)
# 							t_1=time.time()
# 						print 'End of cycle'
# 						cycle+=1
# 					print "End of the test"
#
#
# 				elif input_!= 'N'
# 					print 'Pls enter a valid command'
#
#
#
#
# 			else:
# 				print "pls initialise"
#
#
# newstdin=os.fdopen(os.dup(sys.stdin.fileno()))
