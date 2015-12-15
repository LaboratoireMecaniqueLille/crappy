import serial
import time


class ActuatorLal300(object):
	
	def __init__(self,param,ser): # add param as parameter
		self.param=param
		self.ser=ser

	def reset(self): #Reinitialisation des parametres (gain) du moteur
		self.ser.write("MF,RM,SG%i,SI%i,SD%i,IL%i,DH\r\n"%(self.param['PID_PROP'],self.param['PID_INT'],self.param['PID_DERIV'],self.param['PID_INTLIM'])) #set PID values valeur DH modifiee
		time.sleep(0.005)
		#while self.ser.inWaiting()>0:
			#self.ser.read()
		self.ser.read(self.ser.inWaiting())
		time.sleep(0.005)
		
	def homing(self): #Deplacement au point d'origine "Home"
		self.ser.write("PM,MA-10,GO\r\n")
		time.sleep(0.005)
		#while ser.inWaiting()>0:
			#self.ser.read()
		self.ser.read(self.ser.inWaiting())
		return self.ser

	def stoplal300(self): #Arret du moteur
		self.ser.write("MF\r\n") #Envoi de la commande "Stop"
		time.sleep(0.005)
		#while self.ser.inWaiting()>0:
			#self.ser.read()
		self.ser.read(self.ser.inWaiting())

	def closelal300(self):#Fermeture du port serie
		self.stoplal300()#Arret du moteur
		self.ser.close()  
		return self.ser.isOpen() #verification de l'ouverture du port serie

	def set_position(self,consigne): #Envoi de la consigne de position au moteur
		self.ser.write(consigne) #Ecriture de la consigne	
		time.sleep(0.005)
		#while self.ser.inWaiting()>0:
			#self.ser.read()
		self.ser.read(self.ser.inWaiting())
		time.sleep(0.025) #Temporisation pour permettre l'effacement complet des commandes stockees dans le port serie
	