# -*- coding: utf-8 -*-
##  @addtogroup blocks
# @{

##  @defgroup AutoDrive AutoDrive
# @{

## @file _autoDrive.py
# @brief Command the motor to follow the spots with the camera, takes a gain and a direction
#
# @author Victor Couty
# @version 1.0
# @date 10/02/2017
from __future__ import print_function, division

from _masterblock import MasterBlock

def listify(s):
  return map(float,filter(None,s.lstrip('[ ').rstrip(' ]').split(' ')))

class AutoDrive(MasterBlock):
  """
  This block gets data from videoextenso and drives a technical or an actuator to move the camera in order to keep
  the spots in the center of the frame
  """

  def __init__(self, **kwargs):
    MasterBlock.__init__(self)
    for arg,default in [('technical',None), # If specified, will open crappy.technical.xx as the device to command
			('actuator',None), # Same as above but if it is an actuator, one of these two should be sepcified
			('dev_args', {}), # The kwargs to be given to the technical/actuator
			('P', 500), # The gain for commanding the technical/actuator
			('direction', 'Y-'), # The direction to follow (X/Y +/-), depending on camera orientation
			('range',2048), # The number of pixels in this direction
			]:
      setattr(self,arg,kwargs.get(arg,default))
      try:
        del kwargs[arg]
      except KeyError:
        pass
    if len(kwargs) != 0:
      raise AttributeError("[AutoDrive] Unknown kwarg(s):"+str(kwargs))
    sign = -1 if self.direction[1] == '-' else 1
    self.P *= sign

  def get_class(self):
    """
    This method simply imports the correct technical or actuator and instanciates it, it raises an error 
    if the given device does not exist or no device was specified
    """
    if self.technical is not None:
      i = __import__('crappy.technical',fromlist=[self.technical])
      try:
	return getattr(i,self.technical)
      except AttributeError:
	raise NotImplementedError('[AutoDrive] No such technical:'+self.technical)
    elif self.actuator is not None:
      i = __import__('crappy.actuator',fromlist=[self.actuator])
      try:
	return getattr(i,self.actuator)
      except AttributeError:
	raise NotImplementedError('[AutoDrive] No such actuator:'+self.actuator)
    else:
      raise AttributeError("[AutoDrive] You must specify a technical or an actuator!")

  def get_center(self,data):
    l = listify(data['P'+self.direction[0].lower()])
    #print(l,type(l))
    #print("Center:",(max(l)+min(l))/2)
    return (max(l)+min(l))/2

  def prepare(self):
    self.device = self.get_class()(**self.dev_args)
    if hasattr(self.device,'actuator'): # Most technical have their actuatos methods under self.actuator
      self.device = self.device.actuator
    self.device.set_speed(0) # Make sure it is stopped

  def main(self):
    """
    Apply the command received by a link to the technical object.
    """
    try:
      while True:
	data = self.inputs[0].recv_last() # Get the data
	#print("Diff:",(self.get_center(data)-self.range/2))
	self.device.set_speed(int(self.P*(self.get_center(data)-self.range/2))) # And set speed to P*(img center-spots center)
    except (Exception,KeyboardInterrupt) as e:
      print("[Autodrive] Encountered an exception",e,"stopping actuator!")
      self.device.set_speed(0)
      raise
