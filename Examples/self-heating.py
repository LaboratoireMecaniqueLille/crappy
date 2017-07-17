"""
This example shows an usage of crappy to run some self-heating tests.
It requires two labjacks T7, and you need to fill the "identifier" keyword 
depending on which labjack you use.
"""
import crappy
import time
import numpy as np

directory = '/home/francois/Essais/self-heating-prise2/traction_20pts/'

class ConditionCalib(crappy.links.Condition):
  """
  Used to compute directly self-heating temperature from 3 thermocouples, as 
  described in Munier thesis.
  """

  def __init__(self):
    pass

  def evaluate(self, value):
    calc = value['Tspecimen'] - (value['Tdown'] + value['Tup']) / 2.
    value['Delta'] = calc
    return value

class EvalStress(crappy.links.Condition):
  """
  Used to compute strain stress related to torque applied by the instron, and 
  tensile stress as well.
  """

  def __init__(self):
    self.section = np.pi * (12.5 ** 2 - 11 ** 2)
    self.I = np.pi * (25e-3 ** 4 - 22e-3 ** 4) / 32
    self.rmax = 25e-3 / 2

  def evaluate(self, value):
    value['Shear(MPa)'] = (
      (value['Torque(Nm)'] / self.I) * self.rmax * 10 ** -6)
    value['Stress(MPa)'] = (value['Force(N)'] / self.section)
    return value

labjack_instron = crappy.blocks.IOBlock("Labjack_T7", 
    labels=["time(sec)", "Position(mm)", "Effort(kN)"],
    channels=["AIN0", "AIN1"],
    gain = [0.5, 8]  # mm/V, kN/V
    offset=0,
    chan_range=10,
    make_zero=True,
    resolution=0,
    identifier='ANY')
saver_instron = crappy.blocks.Saver(directory + 'Instron.csv')
crappy.link(labjack_instron, saver_instron, condition=EvalStress())

labels = ["time(sec)", 'Tspecimen', 'Tup', 'Tdown']

labjack_temperatures = crappy.blocks.IOBlock("Labjack_T7",
    mode="thermocouple",
    channels=range(3),
    labels=labels,
    identifier='ANY')
saver_temperatures = crappy.blocks.Saver(directory + 'Temperatures.csv')
grapher_temperatures = crappy.blocks.Grapher([('time(sec)', label) for label in labels[1:]], length=1800)

crappy.link(labjack_temperatures, grapher_temperatures, condition=ConditionCalib())
crappy.link(labjack_temperatures, saver_temperatures, condition=ConditionCalib())
crappy.start()

