# coding: utf-8

from time import time

from .block import Block
from ..actuator import actuator_list


class Machine(Block):
  """
  To drive a machine with a one or more actuators

  Takes a list of dicts, containing the information to create each actuator
  Each key will stand for a parameter (see below), else they are transferred
  to the actuator.
  Args:
    - actuators (list of dict): The list of the actuators of the machine.
      Each dict must have keys according to the list below and the kwargs
      you want to send to the actuator.
    - common (dict, default:{}) The keys of this dict will be added to
      each dict. However, if a same key is in an arg AND in common, the arg
      will prevail.
    - freq (float, default 200): The looping frequency of the block
    - time_label (str, default='t(s)'): If reading data from one or more
      actuators, the time will also be returned under this label.
    - spam (bool, default=False) : If True, the command is sent on each loop,
      else, it is sent every time a value is received

  If you want to forward a parameter to the actuator that has the name of
  a key below, add an underscore at the end of its key
  keys:
    - type (str): The name of the actuator to instanciate
    - cmd (str): The label of the input to drive the axis
    - mode: ('speed'|'position', default='speed')
      Will either call set_speed or set_position on the actuator
    - speed: If mode is position, the speed of the axis
    - pos_label: If set, the block will return the value of .get_position
      with this label
    - speed_label: same as pos_label but with get_speed
  """
  def __init__(self, actuators, common={}, freq=200, time_label='t(s)',
      spam=False):
    Block.__init__(self)
    self.freq = freq
    self.time_label = time_label
    self.spam = spam
    self.settings = [{} for i in actuators]
    for setting, d in zip(self.settings, actuators):
      d.update(common)
      for k in ('type', 'cmd'):
        setting[k] = d[k]
        del d[k]
      if 'mode' in d:
        assert d['mode'].lower() in ('position', 'speed')
        setting['mode'] = d['mode'].lower()
        del d['mode']
        if 'speed' in d:
          setting['speed'] = d['speed']
      else:
        setting['mode'] = 'speed'
      for k in ('pos_label', 'speed_label'):
        if k in d:
          setting[k] = d[k]
          del d[k]
      setting['kwargs'] = d

  def prepare(self):
    self.actuators = []
    for setting in self.settings:
      # Open each actuators with its associated dict of settings
      self.actuators.append(actuator_list[setting['type'].capitalize()](
        **setting['kwargs']))
      self.actuators[-1].open()

  def send_data(self):
    to_send = {}
    for actuator, setting in zip(self.actuators, self.settings):
      if 'pos_label' in setting:
        to_send[setting['pos_label']] = actuator.get_pos()
      if 'speed_label' in setting:
        to_send[setting['speed_label']] = actuator.get_speed()
    if to_send != {}:
      to_send[self.time_label] = time() - self.t0
      self.send(to_send)

  def begin(self):
    self.send_data()

  def loop(self):
    if self.spam:
      recv = self.get_last()
    else:
      recv = self.recv_all()
    for actuator, setting in zip(self.actuators, self.settings):
      if setting['mode'] == 'speed' and setting['cmd'] in recv:
        actuator.set_speed(recv[setting['cmd']])
      elif setting['mode'] == 'position' and setting['cmd'] in recv:
        try:
          actuator.set_position(recv[setting['cmd']], setting['speed'])
        except (TypeError,KeyError):
          actuator.set_position(recv[setting['cmd']])

    self.send_data()

  def finish(self):
    for actuator in self.actuators:
      actuator.stop()
      actuator.close()
