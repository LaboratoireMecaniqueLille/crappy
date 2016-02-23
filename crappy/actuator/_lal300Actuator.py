# coding: utf-8
import serial
import time


class ActuatorLal300(object):
	def __init__(self,param,ser): # Arguments param et ser indiques dans la classe TechnicalLal300
		"""This class contains methods to command the motor of the Lal300.
		You should NOT use it directly, but use the Lal300Technical.
		"""
		self.param=param
		self.ser=ser

	def reset(self): #Reinitialisation des parametres du correcteur PID du moteur et defintion de l'origine moteur"""
		"""Reset PID parameters and re-defined the motor origin"""
		self.ser.write("MF,RM,SG%i,SI%i,SD%i,IL%i,DH\r\n"%(self.param['PID_PROP'],self.param['PID_INT'],self.param['PID_DERIV'],self.param['PID_INTLIM'])) #set PID values valeur DH modifiee
		time.sleep(0.005) #Temporisation assurant la bonne ecriture de la ligne precedente dans le port serie
		self.ser.read(self.ser.in_waiting) # Nettoyage du port serie
		time.sleep(0.005)
		
	def homing(self): #Deplacement au point d'origine "Home"
		"""Go back to the homing point"""
		self.ser.write("PM,MA-10,GO\r\n")
		time.sleep(0.005)
		self.ser.read(self.ser.in_waiting)
		return self.ser

	def stoplal300(self): #Arret du moteur
		"""Stop the motor"""
		self.ser.write("MF\r\n") #Envoi de la commande "Moteur OFF" via le port serie
		time.sleep(0.005)
		self.ser.read(self.ser.in_waiting) 

	def closelal300(self):#Fermeture du port serie
		"""Close serial port"""
		self.stoplal300()#Arret du moteur
		self.ser.close()  #Fermeture du port serie
		return self.ser.isOpen() #Verification de l'ouverture du port serie (True/False)

	def set_position(self,consigne): #Envoi de la consigne de position au moteur
		"""Send a position command to the motor"""
		self.ser.write(consigne) #Ecriture de la consigne dans le port serie	
		time.sleep(0.005)
		self.ser.read(self.ser.in_waiting)
		time.sleep(0.025) #Temporisation pour permettre l'effacement complet des commandes stockees dans le port serie
	