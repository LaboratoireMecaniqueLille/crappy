from _meta import MasterBlock
import time
import pandas as pd
#import Technical_Lal300
from serial import SerialException
from collections import OrderedDict

class CommandLal300(MasterBlock):
	"""Receive a signal and translate it for the Biotens actuator"""
	def __init__(self, TechnicalLal300):
		self.technical=TechnicalLal300
		self.param=self.technical.param

	def main(self):
            try:
                print "Debut du programme"
                self.technical.actuator.reset()
                t0=time.time()
                inputs_var=self.inputs.recv()
                effort=inputs_var['F(N)']
                lvdt=inputs_var['F(N)']
                block=0
                cycle=0
              
                for etape in range(len(self.param['CYCLES'])):
                    block=0
                    print "Etape= :", etape
                    while block < self.param['CYCLES'][etape]:
                        try:
                            
                            ########################### Commande moteur en mode position ###########################
                            
                            #position=self.technical.sensor.checkdisp()
                            #self.technical.actuator.set_position("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['ETIRE'][etape]))
                            #while position <= (self.param['ETIRE'][etape]+self.param['CORR-']): #or position >= (self.param['ETIRE'][etape]+self.param['CORR+']):
                                #print "Top1"
                                #position=self.technical.sensor.checkdisp()
                                
                            ########################### Commande moteur en vitesse avec limite haute en deplacement #####################
                            
                            self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['ENTREE_VERIN']))
                            while lvdt <= (self.param['ETIRE'][etape]+self.param['CORRLVDT-']): #or lvdt >= (self.param['ETIRE'][etape]+self.param['CORRLVDT+']):
                                print "top3"
                                inputs_var=self.inputs.recv()
                                lvdt=inputs_var['F(N)']
                                
                            block+=0.5
                            cycle+=0.5
                            t=time.time()
                            #Array=pd.DataFrame([[t-t0,cycle,position]],columns=['t(s)','cycle','position'])
                            #Array=OrderedDict(zip(['t(s)','cycle','position'],[t-t0,cycle,position]))
                            #Array=pd.DataFrame([[t-t0,cycle]],columns=['t(s)','cycle'])
                            Array=OrderedDict(zip(['t(s)','cycle'],[t-t0,cycle]))
                            try:
                                for output in self.outputs:
                                    output.send(Array)
                            except:
                                print "panda"
                                
                            ########################### Commande moteur en mode position ###########################
                            
                            #position=self.technical.sensor.checkdisp()
                            #self.technical.actuator.set_position("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['COMPRIME'][etape]))
                            #while position >= (self.param['COMPRIME'][etape]+self.param['CORR+']): #or position <= (self.param['COMPRIME'][etape]+self.param['CORR-'])
                                #print "Top2"
                                #position=self.technical.sensor.checkdisp()
                                
                            ########################### Commande moteur vitesse avec limite basse en effort #####################
                                
                            self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['SORTIE_VERIN']))
                            while effort >= 0.08:#or effort <= 0.25:
                                print "top4"
                                inputs_var=self.inputs.recv()
                                effort=inputs_var['F(N)']
    
                            block+=0.5
                            cycle+=0.5
                            t=time.time()
                            #Array=pd.DataFrame([[t-t0,cycle,position]],columns=['t(s)','cycle','position'])
                            #Array=OrderedDict(zip(['t(s)','cycle','position'],[t-t0,cycle,position]))
                            #Array=pd.DataFrame([[t-t0,cycle]],columns=['t(s)','cycle'])
                            Array=OrderedDict(zip(['t(s)','cycle'],[t-t0,cycle]))
                            try:
                                for output in self.outputs:
                                    output.send(Array)
                            except:
                                print "panda2"
                                
                            #print "Nombre de cycles:",cycle
                        
                        except (SerialException) as s:
                            print "SerialException: ",s
                            pass
                        
                        except (ValueError) as v:
                            print "ValueError detectee: ",v
                            pass
                        
                    #cycle+=cycle
                    #print cycle
                                

                self.technical.actuator.stoplal300()
                time.sleep(0.5)
                self.technical.actuator.homing()
                time.sleep(3)
                self.technical.actuator.reset()
                time.sleep(0.5)
                self.technical.actuator.closelal300()
                print "Fin du programme"
                            
            except KeyboardInterrupt as k:
                    print "Programmme interrompu : ", k
                    self.technical.actuator.stoplal300()
                    time.sleep(0.5)
                    self.technical.actuator.homing()
                    time.sleep(3)
                    self.technical.actuator.reset()
                    time.sleep(0.5)
                    self.technical.actuator.closelal300()
    
# try:
			# last_cmd=0
			# self.last_time=self.t0
			# while True:
				# Data=self.inputs[0].recv()
				# cmd=Data['signal'].values[0]
				# if cmd!= last_cmd:
					# TechnicalLal300.actuator.set_position(cmd)
					# last_cmd=cmd
				# t=time.time()
				# if (t-self.last_time)>=0.2:
					# self.last_time=t
						# position=TechnicalLal300.sensor.checkdisp()
					# Array=pd.DataFrame([[t-self.t0,position]],columns=['t(s)','position'])
					# try:
						# for output in self.outputs:
							# output.send(Array)
					# except:
						# pass
				
		# except (Exception,KeyboardInterrupt) as e:
			# print "Exception in CommandSmacLal300 : ", e
				# TechnicalLal300.actuator.stoplal300()
				# TechnicalLal300.actuator.reset()
			# raise