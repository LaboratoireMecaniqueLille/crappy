# coding: utf-8

import crappy

if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Sine',
                                  'amplitude': 20,
                                  'offset': 50,
                                  'freq': 0.02,
                                  'condition': 'delay=100'},),
                                cmd_label='target_ram(%)',
                                freq=30)

  io = crappy.blocks.IOBlock('FakeInOut',
                             cmd_labels='target_ram(%)',
                             labels=('t(s)', 'ram(%)'),
                             freq=30)

  graph = crappy.blocks.Grapher(('t(s)', 'ram(%)'))

  stop = crappy.blocks.StopButton()

  crappy.link(gen, io)
  crappy.link(io, graph)

  crappy.start()
