# coding: utf-8
"""
Base file for tests using videoextensometry and a marker following actuator.
"""

from time import strftime, gmtime
from pathlib import Path
import crappy

save_path = Path(f"data/{strftime('%a %b %d %H_%M_%S', gmtime())}")
out_gain = 1 / 30  # V/mm
gains = [50, 1 / out_gain]  # N/V mm/V

if __name__ == '__main__':

  # The Block acquiring the images and performing video-extensometry
  ve = crappy.blocks.VideoExtenso(camera='XiAPI', display_images=True,
                                  white_spots=False, freq=30)

  # The Block driving the Actuator for following the spots
  auto_drive = crappy.blocks.AutoDriveVideoExtenso(
      actuator={'name': 'SchneiderMDrive23', 'port': '/dev/ttyUSB0'},
      direction='X-')
  crappy.link(ve, auto_drive)

  # The Block driving the extension of the test sample
  gen = crappy.blocks.Generator(
    path=[{'type': 'CyclicRamp', 'condition1': 'Exx(%)>20', 'speed1': 20 / 60,
           'condition2': 'F(N)<.1', 'speed2': -20 / 60, 'cycles': 5}])
  crappy.link(ve, gen)

  # The Block acquiring the force from the load cell
  labjack = crappy.blocks.IOBlock(
    "LabjackT7",
    channels=[{'name': 'AIN0', 'gain': gains[0], 'make_zero': True},
              {'name': 'AIN1', 'gain': gains[1], 'make_zero': True},
              {'name': 'TDAC0', 'gain': out_gain}],
    labels=['t(s)', 'F(N)', 'x(mm)'], cmd_labels=['cmd'])
  crappy.link(labjack, gen)
  crappy.link(gen, labjack)

  # The Blocks displaying the acquired and calculated values in real-time
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  graph_sensors = crappy.blocks.Grapher(('t(s)', 'F(N)'), ('t(s)', 'x(mm)'))
  crappy.link(ve, graph_extenso)
  crappy.link(labjack, graph_sensors, modifier=crappy.modifier.Mean(10))

  # The Blocks saving the recorded data to text files
  rec_extenso = crappy.blocks.Recorder(save_path / "extenso.csv",
                                       labels=['t(s)', 'Exx(%)', 'Eyy(%)'])
  rec_sensors = crappy.blocks.Recorder(save_path / "sensors.csv",
                                       labels=['t(s)', 'F(N)', 'x(mm)'])
  crappy.link(ve, rec_extenso)
  crappy.link(labjack, rec_sensors)

  # Starting the test
  crappy.start()
