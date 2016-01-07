import serial
import time


class SensorLal300(object):
	
	def __init__(self,param,ser): # add param as parameter
		self.param=param
		self.ser=ser

	def checkdisp(self):#Releve de la cutecom position du moteur (deplacement)
		#try:
		#while self.ser.inWaiting()>0:
		a=self.ser.read(self.ser.inWaiting())
		time.sleep(0.015)
		self.ser.write('TP\r\n')
		time.sleep(0.013) #Temporisation pour permettre l'ecriture de la consigne sur le port serie
		c=self.ser.read(4)
		time.sleep(0.013)
		d=self.ser.read(self.ser.inWaiting()-3)
		#self.ser.readline()#Lecture de la derniere ligne du port serie
		#chaine=self.ser.readline()
		#chaine2=chaine.replace('\r\n',"") #Supression des caracteres \r\n pour obtenir uniquement la position
		time.sleep(0.015)
		disp=int(d) #Conversion de la position en string du port serie en integer
		#print disp
		return disp
		#except Exception:
			#pass
		
	