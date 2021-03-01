# -*- coding: utf-8 -*-

from time import time

from .block import Block
from ..actuator import actuator_list


class AutoDrive(Block):
  """
  To follow the spots with videoextenso.

  Note:
    This blocks takes the output of a Videoextenso block and uses the spots
    coordinates to drive an actuator, on which the camera is mounted.

    This will allow the camera to follow the spots along an axis.

    It is simply a P loop: the difference in pixel between the barycenter
    of the spots and the middle of the image along the given axis is multiplied
    by P and set as the speed of the actuator.

  Kwargs:
    - P (float, default: 2000): The gain.
    - direction (two chars string: {X,Y}{+,-}, default: "Y-"): What axis should
      be considered, and in which direction ?
    - range (int, default: 2048): the size in pixel of the image along this
      axis.
    - max_speed (float, default: 200000): The absolute max value to send to the
      actuator.

  """
  def __init__(self, **kwargs):
    Block.__init__(self)
    for arg,default in [('actuator',{'name':'CM_drive'}),
      ('P', 2000), # The gain for commanding the technical/actuator
      # The direction to follow (X/Y +/-), depending on camera orientation
      ('direction', 'Y-'),
      ('range',2048), # The number of pixels in this direction
      ('max_speed',200000) # To avoid loosing spots swhen going to fast
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
    self.labels = ['t(s)','diff(pix)']

  def get_center(self,data):
    l = data['Coord(px)']
    i = 0 if self.direction[0].lower() == 'y' else 1
    l = [x[i] for x in l]
    return (max(l)+min(l))/2

  def prepare(self):
    actuator_name = self.actuator['name']
    self.actuator.pop('name')
    self.device = actuator_list[actuator_name](**self.actuator)
    self.device.open()
    self.device.set_speed(0) # Make sure it is stopped

  def loop(self):
    data = self.inputs[0].recv_last(blocking=True)
    t = time()
    diff = self.get_center(data)-self.range/2
    self.device.set_speed(
        max(-self.max_speed,min(self.max_speed,int(self.P*diff))))
    self.send([t-self.t0,diff])

  def finish(self):
    self.device.set_speed(0)
