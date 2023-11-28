# coding: utf-8

"""
Detailed example showing a video-extensometry-driven tensile test.

This example uses a Labjack T7 board to send the position command to an Instron
5882 tensile test machine. A Ximea camera is used to measure the strain of the
sample and save the images. Different levels of strain are applied and the
sample is relaxed between each step.
"""

import crappy

SPEED = 5  # mm/min
FORCE_GAIN = 100  # N/V
POS_GAIN = 5  # mm/V

if __name__ == "__main__":

  # Paths to give to the Generator Block
  paths = []
  for strain in [.25, .5, .75, 1., 1.5, 2]:
    paths.append({'type': 'Constant',
                  'value': SPEED / 60,
                  'condition': 'Exx(%)>{}'.format(strain)})
    paths.append({'type': 'Constant',
                  'value': -SPEED / 60,
                  'condition': 'F(N)<1'})

  # The Block generating the signal driving the tensile test machine
  gen = crappy.blocks.Generator(paths, cmd_label='cmd')

  # Arguments for the channels of the Labjack
  force = {'name': 'AIN0', 'gain': FORCE_GAIN}
  pos = {'name': 'AIN1', 'gain': POS_GAIN, 'make_zero': True}
  cmd = {'name': 'TDAC0', 'gain': POS_GAIN}

  # The Block driving the Labjack
  daq = crappy.blocks.IOBlock('LabjackT7', channels=[force, pos, cmd],
                              labels=['t(s)', 'F(N)', 'Position(mm)'],
                              cmd_labels=['cmd'])
  crappy.link(gen, daq)
  crappy.link(daq, gen)

  # This Block records the data acquired by the Labjack
  rec_daq = crappy.blocks.Recorder('results_daq.csv')
  crappy.link(daq, rec_daq)

  # This Block calculates the extension using video-extensometry
  ve = crappy.blocks.VideoExtenso('XiAPI',
                                  save_images=True,
                                  save_folder='img/')
  crappy.link(ve, gen)

  # This Block records the data acquired by the VideoExtenso Block
  rec_ve = crappy.blocks.Recorder('results_ve.csv')
  crappy.link(ve, rec_ve)

  # These Blocks plot the acquired data
  graph_f = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  graph_s = crappy.blocks.Grapher(('t(s)', 'Exx(%)'))
  crappy.link(daq, graph_f, modifier=crappy.modifier.Mean(10))
  crappy.link(ve, graph_s)

  # Starting the test
  crappy.start()
