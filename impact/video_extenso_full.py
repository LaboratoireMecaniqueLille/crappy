# coding: utf-8
"""
Base file for tests using videoextensometry and a marker following actuator.
"""

from time import ctime
import crappy

out_gain = 1 / 30  # V/mm
gains = [50, 1 / out_gain]  # N/V mm/V

timestamp = ctime()[:-5].replace(" ", "_").replace(":", "_")
save_path = "./" + timestamp + "/"

# VideoExtenso and Autodrive blocks
ve = crappy.blocks.Video_extenso(camera='Ximea_cv', show_image=True,
                                 white_spots=False, max_fps=30)

ad = crappy.blocks.AutoDrive(
    actuator={'name': 'CM_drive', 'port': '/dev/ttyUSB0'}, direction='X-')

graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

rec_extenso = crappy.blocks.Recorder(save_path+"extenso.csv",
                                     labels=['t(s)', 'Exx(%)', 'Eyy(%)'])

# Linking them
crappy.link(ve, graph_extenso)
crappy.link(ve, rec_extenso)
crappy.link(ve, ad)

# Labjack
lj = crappy.blocks.IOBlock("Labjack_t7", channels=[
  {'name': 'AIN0', 'gain': gains[0], 'make_zero':True},
  {'name': 'AIN1', 'gain': gains[1], 'make_zero':True},
  {'name': 'TDAC0', 'gain': out_gain}],
                           labels=['t(s)', 'F(N)', 'x(mm)'],
                           cmd_labels=['cmd'])

# Graph
graph_sensors = crappy.blocks.Grapher(('t(s)', 'F(N)'), ('t(s)', 'x(mm)'))
crappy.link(lj, graph_sensors, modifier=crappy.modifier.Mean(10))

# Generator
g = crappy.blocks.Generator(path=[
  {'type': 'cyclic_ramp', 'condition1': 'Exx(%)>20',
   'speed1': 20 / 60, 'condition2': 'F(N)<.1', 'speed2': -20 / 60,
   'cycles': 5}, ])
rec_sensors = crappy.blocks.Recorder(save_path + "sensors.csv",
                                     labels=['t(s)', 'F(N)', 'x(mm)'])

# Linking the generator to all the blocks
crappy.link(ve, g)
crappy.link(lj, g)
crappy.link(g, lj)
crappy.link(lj, rec_sensors)

crappy.start()
