# coding: utf-8

"""
This program is often used as the starting point when performing tests on the
"Biotens" machine.

It creates a new folder for each experiment and performs tensile tests using
videoextensometry.
"""

from time import strftime, gmtime
from pathlib import Path
import crappy

save_path = Path(f"biotens_data/{strftime('%a %b %d %H_%M_%S', gmtime())}")

if __name__ == '__main__':

  # Quick hack for resetting the position of the clamps
  biotens_init = crappy.actuator.Biotens()
  biotens_init.open()
  biotens_init.reset_position()
  biotens_init.set_position(5, 50)

  # The Block providing the signal for driving the tensile test machine
  generator = crappy.blocks.Generator([{'type': 'constant',
                                        'condition': 'F(N)>90',
                                        'value': 5}], freq=100)

  # The Block acquiring the force from the load cell
  effort = crappy.blocks.IOBlock("Comedi", channels=[0], gain=[-48.8],
                                 labels=['t(s)', 'F(N)'])
  # This link enables feedback from the setup to the Generator
  crappy.link(effort, generator)

  # The Block driving the tensile test machine
  biotens = crappy.blocks.Machine([{'type': 'biotens',
                                    'port': '/dev/ttyUSB0',
                                    'position_label': 'position1',
                                    'cmd_label': 'cmd'}])
  crappy.link(generator, biotens)

  # The Block acquiring images from the setup and performing video-extensometry
  extenso = crappy.blocks.VideoExtenso(camera="XiAPI")

  # The Blocks saving the recorded data to text files
  rec_effort = crappy.blocks.Recorder(save_path / "effort.csv")
  rec_position = crappy.blocks.Recorder(save_path / 'position.csv')
  rec_extenso = crappy.blocks.Recorder(save_path / 'extenso.csv',
                                       labels=['t(s)', 'Exx(%)', 'Eyy(%)'])
  crappy.link(effort, rec_effort)
  crappy.link(biotens, rec_position)
  crappy.link(extenso, rec_extenso)

  # The Graphers displaying the acquired or calculated values in real-time
  graph_effort = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(effort, graph_effort)
  crappy.link(extenso, graph_extenso)

  crappy.start()
