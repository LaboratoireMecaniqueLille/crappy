# This is a brand new block to make some self-heating tests.
# Devices needed:
# - Labjack T7, for acquiring instron (stress and positions) and command it
# - LabJack T7 Pro, for acquiring thermocouples data

import time
import numpy as np
import crappy
import os

directory = '/home/francois/Essais/007_self_heating_prise2/Resultats_derive_ndetemps/'

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances


class ConditionCalib(crappy.links.MetaCondition):
    """
    This class is used to acquire temperatures from 3 thermocouples
    """

    def __init__(self):
        pass

    def evaluate(self, value):
        T_specimen = value['Tspecimen']
        T_down = value['Tdown']
        T_up = value['Tup']
        calc = T_specimen - (T_down + T_up) / 2.
        value.update({'Delta': calc})
        return value
#
#
# class EvalStress(crappy.links.MetaCondition):
#     """
#     This class returns strain stress related to torque applied by the instron.
#     """
#
#     def __init__(self):
#         self.I = np.pi * ((25 * 10 ** -3) ** 4 - (22 * 10 ** -3) ** 4) / 32
#         self.rmax = 25 * 10 ** (-3) / 2
#
#     def evaluate(self, value):
#         value['tau(MPa)'] = ((value['C(Nm)'] / self.I) * self.rmax) * 10 ** -6
#         return value


def eval_offset(device, duration):
    timeout = time.time() + duration  # duration secs from now
    print 'Measuring offset (%d sec), please be patient...' % duration
    offset_channels = [[] for i in xrange(device.nb_channels)]
    offsets = []
    while True:
        mesures = device.get_data()[1]
        for i in xrange(len(offset_channels)):
            offset_channels[i].append(mesures[i])

        if time.time() > timeout:
            for i in xrange(len(offset_channels)):
                offsets.append(-np.mean(offset_channels[i]))
            print 'offsets:', offsets
            break
    return offsets

# TEMPERATURES

sensor_thermocouples = crappy.technical.LabJack(sensor={'channels': [0, 1, 2, 3, 4, 5], 'mode': 'thermocouple',
                                                        'resolution': 8}) # 'identifier': '470012790'

labels = ['t(s)', 'Tdown', 'Tup', 'Tspecimen', 'Tair', 'Tdowner', 'Tupper']
measures_temperatures = crappy.blocks.MeasureByStep(sensor_thermocouples,
                                                    labels=labels)


compacter_temperatures = crappy.blocks.Compacter(2)

saver_temperatures = crappy.blocks.Saver(directory + 'Temperatures.csv', stamp='date')

grapher_temperatures = crappy.blocks.Grapher([('t(s)', x) for x in labels[1:]], length=100)
canvas = crappy.blocks.CanvasDrawing(mode='selfheating', bg_image=os.path.realpath('../data/mors_ttc.png'), colormap_range=[25, 40])
# Links

crappy.link(measures_temperatures, compacter_temperatures, condition=ConditionCalib())
crappy.link(compacter_temperatures, saver_temperatures)
crappy.link(compacter_temperatures, grapher_temperatures)
crappy.link(compacter_temperatures, canvas)

# INSTRON

# device_instron = crappy.technical.LabJack(
#     sensor={'channels': ['AIN0', 'AIN1'], 'gain': [1, 1], 'offset': [0, 0], 'chan_range': 10,
#             'resolution': 8, 'identifier': '470012991'},
#     actuator={'channel': 'TDAC0', 'gain': 1, 'offset': 0})
#
# offsets = eval_offset(device_instron, 5)
# device_instron.close()
# device_instron = crappy.technical.LabJack(
#     sensor={'channels': ['AIN0', 'AIN1'], 'gain': [1, 1], 'offset': offsets, 'chan_range': 10,
#             'resolution': 8, 'identifier': '470012991'},
#     actuator={'channel': 'TDAC0', 'gain': 1, 'offset': 0})
# measures_effort = crappy.blocks.MeasureByStep(device_instron, labels=['t(s)', 'Force_measured(N)', 'Position(mm)'])
# compacter_effort = crappy.blocks.Compacter(10)
# save_effort = crappy.blocks.Saver(directory + 'Instron.csv', stamp='date')
# graph_effort = crappy.blocks.Grapher(('t(s)', 'Force_measured(N)'), length=100)  # Add ['t(s)', 'Force_command(N)']
#
# out_measurebystep_effort = crappy.links.Link(name='from_measurebystep_effort')
#
# measures_effort.add_output(out_measurebystep_effort)
# compacter_effort.add_input(out_measurebystep_effort)
#
# out_compacter_effort1 = crappy.links.Link(name='from_compacter_effort1')
# out_compacter_effort2 = crappy.links.Link(name='from_compacter_effort2')
#
# compacter_effort.add_output(out_compacter_effort1)
# compacter_effort.add_output(out_compacter_effort2)
# save_effort.add_input(out_compacter_effort1)
# graph_effort.add_input(out_compacter_effort2)

# Starting objects

crappy.start()
