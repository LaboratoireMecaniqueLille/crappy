#coding: utf-8

from .masterblock import MasterBlock
from ..actuator import actuator_list

class Machine(MasterBlock):
  def __init__(self, actuators,common={},freq=200):
    """
    Takes a list of dicts, containing the information to create each actuator
    Each key will stand for a parameter (see below), else they are transferred
    to the actuator.
    The content of the dict common will be added to each dict
    However, if a same key is in an arg AND in common, the arg will prevail
    If you want to forward a parameter to the actuator that has the name of
    a key below, add an underscore at the end of its key
    keys:
      type: The name of the actuator to instanciate
        Mandatory
      cmd: The label of the input to drive the axis
        Mandatory
      mode: 'speed'|'position'
        Will either call set_speed or set_position on the actuator
        Default: speed
      pos_label: If set, the block will return the value of .get_position
        with this label
      speed_label: same as pos_label but with get_speed
    """
    MasterBlock.__init__(self)
    self.freq = freq
    self.settings = [{} for i in actuators]
    for setting,d in zip(self.settings,actuators):
      d.update(common)
      for k in ('type','cmd'):
        setting[k] = d[k]
        del d[k]
      if 'mode' in d:
        assert d['mode'].lower() in ('position','speed')
        setting['mode'] = d['mode'].lower()
        del d['mode']
      else:
        setting['mode'] = 'speed'
      for k in ('pos_label','speed_label'):
        if k in d:
          setting[k] = d[k]
          del d[k]
      setting['kwargs'] = d

  def prepare(self):
    self.actuators = []
    for setting in self.settings:
      self.actuators.append(actuator_list[setting['type'].capitalize()](
        **setting['kwargs']))
      self.actuators[-1].open()

  def loop(self):
    recv = self.get_last()
    to_send = {}
    for actuator,setting in zip(self.actuators,self.settings):
      if setting['mode'] == 'speed':
        actuator.set_speed(recv[setting['cmd']])
      elif setting['mode'] == 'position':
        actuator.set_speed(recv[setting['cmd']])
      if 'pos_label' in setting:
        to_send[setting['pos_label']] = actuator.get_pos()
      if 'speed_label' in setting:
        to_send[setting['speed_label']] = actuator.get_speed()
    if to_send != {}:
      self.send(to_send)

  def finish(self):
    for actuator in self.actuators:
      actuator.close()
