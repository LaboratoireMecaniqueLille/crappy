# This is a brand new block to make some self-heating tests.
# Devices needed:
# - Labjack T7, for acquiring instron (stress and positions) and command it
# - LabJack T7 Pro, for acquiring thermocouples data

import time
import numpy as np
import crappy
import os

directory = '/home/francois/Essais/007_self_heating_prise2/Resultats_derive_ndetemps/'


class ConditionCalib(crappy.links.Condition):
  """
  This class is used to acquire temperatures from 3 thermocouples
  """

  def __init__(self):
    pass

  def evaluate(self, value):
    calc = value['T_specimen'] - (value['T_down'] + value['T_up']) / 2.
    value['Delta'] = calc
    return value


class EvalStress(crappy.links.Condition):
  """
  This class returns strain stress related to torque applied by the instron.
  """

  def __init__(self):
    self.I = np.pi * (25e-3 ** 4 - 22e-3 ** 4) / 32
    self.rmax = 25e-3 / 2

  def evaluate(self, value):
    value['tau(MPa)'] = (value['C(Nm)'] / self.I) * self.rmax * 10 ** -6
    return value


def eval_offset(device, duration):
  timeout = time.time() + duration  # duration secs from now
  print 'Measuring offset (%d sec), please be patient...' % duration
  offset_channels = [[] for i in xrange(device.nb_channels)]
  offsets = []
  while True:
    mesures = device.get_data("all")[1]
    print"mesures:", mesures
    for i in xrange(len(offset_channels)):
      offset_channels[i].append(mesures[i])

    if time.time() > timeout:
      for i in xrange(len(offset_channels)):
        offsets.append(-np.mean(offset_channels[i]))
      print 'offsets:', offsets
      break
  return offsets


sensor_thermocouples = crappy.technical.LabJack(
  sensor={'channels': [0, 1, 2, 3],
          'mode': 'thermocouple',
          'resolution': 8})

labels = ['t(s)', 'Tspecimen', 'Tup', 'Tdown','Tair']
measures_temperatures = crappy.blocks.MeasureByStep(sensor_thermocouples,
                                                    labels=labels)
saver_temperatures = crappy.blocks.Saver(directory + 'Temperatures.csv',
                                         stamp='date')
grapher_temperatures = crappy.blocks.Grapher([('t(s)', x) for x in labels[1:]],
                                             length=100)
canvas = crappy.blocks.CanvasDrawing(drawing='selfheating',
                                     bg_image=os.path.realpath(
                                       '../data/mors_ttc.png'),
                                     colormap_range=[25, 40])
# Links

crappy.link(measures_temperatures, saver_temperatures)
crappy.link(measures_temperatures, grapher_temperatures)
crappy.link(measures_temperatures, canvas)

# INSTRON
comedi_dict = {'device': '/dev/comedi0',
               'channels': [0, 1, 3, 4],
               'gain': [1.5, 2e4, 10, 500],
               'offset': 0}
comedi_labels = ['time(sec)', 'Force(N)', 'Position(mm)', 'Rotation(deg)',
                 'Couple(Nm)']
comedi_instron = crappy.sensor.ComediSensor(**comedi_dict)
comedi_dict['offset'] = eval_offset(comedi_instron, 1)
comedi_instron.close()
comedi_instron = crappy.sensor.ComediSensor(**comedi_dict)
measures_effort = crappy.blocks.MeasureByStep(labels=comedi_labels)
save_effort = crappy.blocks.Saver(directory + 'Instron.csv', stamp='date')
graph_effort = crappy.blocks.Grapher(('t(s)', 'Force(N)'),
                                     length=100)  # Add ['t(s)', 'Force_command(N)']
dashboard = crappy.blocks.Dashboard(nb_digits=3)

crappy.link(measures_effort, save_effort)
crappy.link(measures_effort, graph_effort)
crappy.link(measures_effort, dashboard)

crappy.start()
