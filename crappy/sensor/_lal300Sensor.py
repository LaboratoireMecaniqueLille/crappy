# coding: utf-8
import serial
import time


class SensorLal300(object):
    def __init__(self,param,ser):# Arguments param et ser indiques dans la classe TechnicalLal300
        """This class contains methods to get info from the motors of the lal300
        machine. You should NOT use it directly, but use the Lal300Technical.
        """
        self.param=param
        self.ser=ser

    def checkdisp(self):#Releve de la position du moteur via le port serie
        """Check current position."""
        self.ser.read(self.ser.in_waiting) #Nettoyage du port serie
        time.sleep(0.015)
        self.ser.write('TP\r\n') #Ecriture de l'instruction "Tell Position" pour indiquer la position moteur
        time.sleep(0.013) #Temporisation pour permettre l'ecriture de la consigne sur le port serie
        self.ser.read(4) #Lire et effacer les 4 caracteres donnes par l'instruction precedente
        time.sleep(0.013)
        d=self.ser.read(self.ser.in_waiting-3) #Lire et effacer les tous caracteres donnes par l'instruction precedente sauf les 3 derniers correspondant au \r \n et au retour a la ligne
        time.sleep(0.015)
        disp=int(d) #Conversion en integer de la position lue via le port serie
        return disp
