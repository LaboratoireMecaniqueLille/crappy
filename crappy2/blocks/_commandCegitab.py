#!/usr/bin/python
# -*- coding: utf-8 -*-
#Cegitab serial ports

from crappy2.blocks import MasterBlock
import sys
import glob
import serial
import io
import time
from multiprocessing import *
import pandas as pd
from collections import OrderedDict 

#Désignation		Win	Port name	Baudrate	ByteSize	Parity
#Actionneur vérin	COM5	'/dev/ttyUSB0'	19200	
#Capteur effort		COM4	'dev/ttyS0'	9600		7		Odd
#Platine XY - X		COM4	'/dev/ttyUSB1'	19200
#Platine XY - Y		COM3	'/dev/ttyUSB2'	19200
class SerialPortActuator:
  def __init__(self,port,ConversionFactor):
      print "commandcegitab !"   
      #Actuator _ Declaration
      try:
	self.port = serial.Serial(port)
	self.port.timeout = 0.01
	self.port.baudrate = 19200
	self.port.bytesize = 8
	self.port.stopbits = 1
	self.port.parity = 'N'
	self.port.xonxoff = False
	self.port.rtscts = False
	self.port.dsrdtr = False
	self.port.close()
	self.port.open()
	print(self.port)
	self.port.close()
      except:
	pass
      self.ConversionFactor = ConversionFactor

  #Declaration
  def OM_Command(self,Commande):
    Commande = Commande + chr(10)
    return Commande

  #ClearAlarm
  def ClearAlarm(self):
    self.port.open()
    print('ALMCLR')
    self.port.write(self.OM_Command('ALMCLR'))
    self.port.close()

  #MoveUp
  def MoveUp(self):
    self.port.open()
    print('UP')
    self.port.write(self.OM_Command('MCN'))
    self.port.close()
    
  #MoveDown
  def MoveDown(self):
    self.port.open()
    print('DOWN')
    self.port.write(self.OM_Command('MCP'))
    self.port.close()
    
  #MoveStop
  def MoveStop(self):
    self.port.open()
    print('STOP')
    self.port.write(self.OM_Command('SSTOP'))
    self.port.close()

  #Speed (conversion from step/s to mm/s)
  def Speed(self,Speed):
    SpeedInc = float(Speed)/self.ConversionFactor
    Commande = 'VR ' + str(SpeedInc)
    self.port.open()
    print('ActuatorRealSpeed', Speed)
    self.port.write(self.OM_Command(Commande))
    self.port.close()

  #Position (conversion from step/s to mm/s)
  def Position(self):
    self.port.open()
    self.port.write(self.OM_Command('PC'))
    self.port.flushInput()
    self.port.write(self.OM_Command('PC'))
    a_jeter=self.port.readline()
    ActuatorPos = self.port.readline()
    self.port.close()
    ActuatorPos = str(ActuatorPos)
    ActuatorPos = ActuatorPos[4::]
    ActuatorPos = ActuatorPos[::-1]
    ActuatorPos = ActuatorPos[3::]
    ActuatorPos = ActuatorPos[::-1]
    try:
      ActuatorPos = float(ActuatorPos)*self.ConversionFactor
      return ActuatorPos
    except ValueError:
      print "PositionReadingError" #Se prévenir des erreur de lecture quand la readline fait de la merde
    #return 12.0
  def TarePosition(self):
    self.port.open()
    print('TarePosition')
    self.port.write(self.OM_Command('preset'))
    self.port.close()
    

class SerialPortCaptor:
  def __init__(self,port):
    
      "Capteur d'effort _ Declaration"
      self.port = serial.Serial(port)
      self.timeout = 1
      self.baudrate = 9600
      self.bytesize = 7
      self.stobits = 1
      self.parity = 'O'
      self.xonxoff = False
      self.rtscts = False
      self.dsrdtr = False
      print(self.port)
      self.port.close()

  "Capteur effort _ Fonction lecture brutte"
  def Read(self):
    self.port.open()
    self.port.write('TA*')
    ReadLine = self.port.readline()
    self.port.close()
    ReadLine = str(ReadLine)
    ReadLine = ReadLine[::-1]
    ReadLine = ReadLine[2::]
    ReadLine = ReadLine[::-1]
    ReadLine = float(ReadLine)
    try:
      ReadLine = float(ReadLine)
      return ReadLine
    except ValueError:
      print "LoadReadingError" #Se prévenir des erreur de lecture quand la readline fait de la merde
    #return 200.0

class PipeCegitab(MasterBlock):
  def __init__(self,Actuator,Load):
    print "pipecegitab !"
    super(PipeCegitab, self).__init__()
    self.Actuator=Actuator
    self.Load=Load
  def main(self):
    try:
      condition = 1
      while condition == 1:
	time.sleep(0.1)
	pos=self.Actuator.Position()
	F=self.Load.Read()
	#Array=pd.DataFrame([[time.time()-self.t0,pos,F]],columns=['t(s)','position(mm)','F(N)'])
	Array=OrderedDict(zip(['t(s)','position(mm)','F(N)'],[time.time()-self.t0,pos,F]))
	try:
	  for output in self.outputs:
	    output.send(Array)
	except KeyboardInterrupt:
	  raise
	except Exception as e:
	  print e
	  pass
    except KeyboardInterrupt:
      print "KeyboardInterrupt received in PipeCegitab"
      raise
    except Exception as e:
      print e
      #else:
	#print "Hasta la vista"
