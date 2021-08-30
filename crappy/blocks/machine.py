# coding: utf-8

from time import time

from .block import Block
from ..actuator import actuator_list


class Machine(Block):
  """To drive a machine with a one or more :ref:`Actuators`.

  Takes a :obj:`list` of :obj:`dict`, containing the information to create each
  actuator. Each key will either stand for a parameter, or else they are
  transferred to the actuator.
  """

  def __init__(self,
               actuators,
               common=None,
               freq=200,
               time_label='t(s)',
               spam=False):
    """Sets the args and initializes the parent class.

    Args:
      actuators (:obj:`list`): The :obj:`list` of the :ref:`Actuators` of the
        machine. It contains one or several :obj:`dict`, whose mandatory keys
        are described below. The other keys will be passed to the actuator as
        arguments.
      common (:obj:`dict`, optional): The keys of this :obj:`dict` will be
        common to all of the actuators. However if this conflicts with an
        already existing key for an actuator, the latter will prevail.
      freq (:obj:`float`, optional): The looping frequency of the block.
      time_label (:obj:`str`, optional): If reading data from one or more
        actuators, the time will be returned under this label.
      spam (:obj:`bool`, optional): If :obj:`True`, a command is sent on each
        loop of the block, else it is sent every time a value is received.

    Note:
      - ``actuators`` keys:

        - ``type`` (:obj:`str`): The name of the actuator to instantiate.
        - ``cmd`` (:obj:`str`): The label of the input to drive the axis.
        - ``mode`` (:obj:`str`, default: `'speed'`): Can be either `'speed'` or
          `'position'`. Will either call :meth:`set_speed` or
          :meth:`set_position` to drive the actuator.
        - ``speed`` (:obj:`float`): If mode is `'position'`, the speed of the
          axis.
        - ``pos_label`` (:obj:`str`): If set, the block will return the value
          of :meth:`get_position` with this label.
        - ``speed_label`` (:obj:`str`): If set, the block will return the value
          of :meth:`get_speed` with this label.
    """

    Block.__init__(self)
    if common is None:
      common = {}
    self.freq = freq
    self.time_label = time_label
    self.spam = spam
    self.settings = [{} for _ in actuators]
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
        to_send[setting['pos_label']] = actuator.get_position()
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
      recv = self.recv_all_last()
    for actuator, setting in zip(self.actuators, self.settings):
      if setting['mode'] == 'speed' and setting['cmd'] in recv:
        actuator.set_speed(recv[setting['cmd']])
      elif setting['mode'] == 'position' and setting['cmd'] in recv:
        try:
          actuator.set_position(recv[setting['cmd']], setting['speed'])
        except (TypeError, KeyError):
          actuator.set_position(recv[setting['cmd']])

    self.send_data()

  def finish(self):
    for actuator in self.actuators:
      actuator.stop()
      actuator.close()
