# coding: utf-8

from _meta import MasterBlock
import time
import pandas as pd
from serial import SerialException
from collections import OrderedDict
import numpy as np
#from time import *

class CommandLal300(MasterBlock):
    """Programme permettant de realiser des essais cycliques avec parametres de deplacement, de cycles et de vitesse variables"""	
    def __init__(self, TechnicalLal300): #Initialisation de la classe CommandLal300 avec la classe TechnicalLal300
            """
Parameters
----------
TechnicalLal300 : crappy.technical.TechnicalLal300 object.
            """
            self.technical=TechnicalLal300
            self.param=self.technical.param

    def main(self):
        try:
            t10=time.time()
            for q in range (52):
                time.sleep(3600)
                temps=(time.time()-t10)/3600.0
                print "temps: ", temps
            #tabs=localtime()
            print("Debut du programme SMAC:") #+str(tabs.tm_hour)+"h "+str(tabs.tm_min)+"mn ")
            self.technical.ser.write("DH\r\n")#Initialisation des parametre du correcteur PID et de la position 0 de la traverse mobile
            time.sleep(0.1)
            print "Remise a zero"
            block=0 # Initialisation des variables 
            cycle=0
            etape=0
            print "Initialisation des variables"
            
            while etape < len(self.param['CYCLES']): ## Pour le mode sans limite en effort
                block=0
                print "Etape= :", etape
                while block < self.param['CYCLES'][etape]: #Block= cycles locaux, reinitialise a chaque franchissement d'etapes
                    try:
                        ########################### Commande moteur en mode position: Traction ###########################
                        
                        #position=self.technical.sensor.checkdisp()
                        inputs_var=self.inputs[0].recv() #recuperation des variables en entree du block CommandLal300
                        position=(-10000*inputs_var['dep(mm)']) #definition d'une variable position (en unite moteur) via le deplacement LVDT en mm
                        while self.technical.ser.in_waiting > 0:
                            self.technical.ser.read(1)
                        self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,%s,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['ENTREE_VERIN'])) #Ecriture de la consigne en mode vitesse avec les parametres adequats (cf documentation moteur SMAC Lal 300)
                        while position >= (self.param['ETIRE'][etape]): #Comparaison entre variable position et param ETIRE de l'etape en cours lorsque condition vraie: si position=ETIRE sortie de boucle
                            inputs_var=self.inputs[0].recv()
                            position=(-10000*inputs_var['dep(mm)'])
                            if position < -180000:
                                break
                            
                        while self.technical.ser.in_waiting > 0:
                            self.technical.ser.read(1)
                            
                        self.technical.actuator.set_position("MN,VM,SA0,SV0,SQ30000,DI0,GO\r\n")
                        time.sleep(0.1)
                        
                        block+=0.5 #Incrementation des cycles locaux et cycles globaux
                        cycle+=0.5
                        t=time.time()
                        s=t-self.t0
                        
                        Array=OrderedDict(zip(['t(s)','cycle','position'],[s,cycle,position])) #Stockage des variables dans une matrice
                
                        try:
                            for output in self.outputs:
                                output.send(Array) #Envoi de la matrice des variables en sortie du block CommandLal300
                                
                        except :
                                print "Erreur envoi Array1"
                            
                        ########################### Commande moteur en mode position: Compression ###########################

                        #position=self.technical.sensor.checkdisp()
                        inputs_var=self.inputs[0].recv()
                        position=(-10000*inputs_var['dep(mm)'])
                        while self.technical.ser.in_waiting > 0:
                            self.technical.ser.read(1)
                        self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,%s,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['SORTIE_VERIN']))
                        while position <= (self.param['COMPRIME'][etape]):
                            inputs_var=self.inputs[0].recv()
                            position=(-10000*inputs_var['dep(mm)'])
                            if position > 400:
                                break
                            
                        while self.technical.ser.in_waiting > 0:
                            self.technical.ser.read(1)
                        
                        self.technical.actuator.set_position("MN,VM,SA0,SV0,SQ30000,DI0,GO\r\n")
                        time.sleep(0.1)
                        #print "Vitesse nulle comprime"

                        block+=0.5
                        cycle+=0.5
                        t=time.time()
                        s=t-self.t0
                        
                        Array=OrderedDict(zip(['t(s)','cycle','position'],[s,cycle,position]))

                        try:
                            for output in self.outputs:
                                output.send(Array)
                        except:
                            print "Erreur envoi Array2"
                            
                        if cycle%2000==0:
                            print "Nombre de cycles: ", cycle

                    
                    except (SerialException) as s: #Gestion des exceptions SerialException et ValueError lors d'une mauvaise lecture du port serie: le programme alerte et poursuit
                        print "SerialException: ",s
                        pass
                    
                    except (ValueError) as v:
                        print "ValueError detectee: ",v
                        pass
                    
                etape+=1
            print("Fin du programme SMAC:") #+str(tabs.tm_hour)+"h "+str(tabs.tm_min)+"mn ")
            self.technical.actuator.stoplal300()
            time.sleep(0.5)
            #self.technical.actuator.homing()
            #time.sleep(3)
            #self.technical.actuator.reset()
            #time.sleep(0.5)
            self.technical.actuator.closelal300()
        
                        
        except KeyboardInterrupt as k:
                print "Programmme interrompu : ", k
                self.technical.actuator.stoplal300()
                time.sleep(0.5)
                #self.technical.actuator.homing()
                #time.sleep(3)
                #self.technical.actuator.reset()
                #time.sleep(0.5)
                self.technical.actuator.closelal300()
                
        finally:
            print "Programmme interrompu : "
            self.technical.actuator.stoplal300()
            time.sleep(0.5)
            #self.technical.actuator.homing()
            #time.sleep(3)
            #self.technical.actuator.reset()
            #time.sleep(0.5)
            self.technical.actuator.closelal300()
                    
    
    
                #while etape < len(self.param['CYCLES']): ### Pour le mode avec limite en effort
                    #block=0
                    #print "Etape= :", etape
                    #while block < self.param['CYCLES'][etape]:
                        #try:
                            ############################ Commande moteur en vitesse avec limite haute en deplacement #####################
                            
                            #inputs_var=self.inputs[0].recv()
                            #position=(-10000*inputs_var['dep(mm)'])
                            #self.technical.ser.read(self.technical.ser.inWaiting())
                            #time.sleep(0.13)
                            #self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,%s,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['ENTREE_VERIN']))
                            #while position >= (self.param['ETIRE'][etape]):
                                #inputs_var=self.inputs[0].recv()
                                #position=(-10000*inputs_var['dep(mm)'])
                                #print "position3 =",position
                                
                            #block+=0.5
                            #cycle+=0.5
                            #t=time.time()
                            #s=t-t0
                
                            #Array=OrderedDict(zip(['t(s)','cycle','position'],[s,cycle,position])) ####### Mode position
   
                            #try:
                                #for output in self.outputs:
                                    #output.send(Array)
                                    
                            #except :
                                 #print "panda"
                                 
                            ############################ Commande moteur vitesse avec limite basse en effort #####################
                                
                            #inputs_var=self.inputs[0].recv()
                            #effort=inputs_var['F(N)']
                            #position=(-10000*inputs_var['dep(mm)'])
                            #self.technical.ser.read(self.technical.ser.inWaiting())
                            #time.sleep(0.13)   
                            #self.technical.actuator.set_position("MN,VM,SA%i,SV%i,SQ%i,%s,GO\r\n"%(self.param['ACC'],self.param['SPEED'][etape],self.param['FORCE'],self.param['SORTIE_VERIN']))
                            #while effort >= (effort*0.2):
                                #if position >= -100:
                                    #break
                                #inputs_var=self.inputs[0].recv()
                                #position=(-10000*inputs_var['dep(mm)'])
                                #effort=inputs_var['F(N)']
                                #print "effort=", effort
                                #print "position4 =",position
    
                            #block+=0.5
                            #cycle+=0.5
                            #t=time.time()
                            #s=t-t0
                            
                            #Array=OrderedDict(zip(['t(s)','cycle','position'],[s,cycle,position]))
                            
                            #try:
                                #for output in self.outputs:
                                    #output.send(Array)
                            #except:
                                #print "panda2"
                                
                            #if cycle%1==0:
                                #print "Nombre de cycles: ", cycle
    
                        
                        #except (SerialException) as s:
                            #print "SerialException: ",s
                            #pass
                        
                        #except (ValueError) as v:
                            #print "ValueError detectee: ",v
                            #pass
                        
                    #etape+=1
    