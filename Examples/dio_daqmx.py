# coding: utf-8

"""
Example demonstrating the use of National instrument daq boards.

Required hardware:
  - NI DAQ board with at least one analog input and one digital input
    compatible with the nidaqmx module.
"""

import crappy
from functools import partial


def intify(data):
  """Used to replace False by 0 and True by 1 in the digital inputs.

  Not doing so results in an error in the grapher.
  """

  for i, d in enumerate(data):
    if isinstance(d, bool):
      data[i] = int(d)
  return data


def change_name(data, prev: str, new: str):
  """Modifier for changing the name of a label."""

  val = data[prev]
  del data[prev]
  data[new] = val

  return data


if __name__ == "__main__":
  gen = crappy.blocks.Generator([dict(type='cyclic', value1=0, value2=1,
                                      condition1="delay=1",
                                      condition2="delay=1")], repeat=True)
  io = crappy.blocks.IOBlock("Nidaqmx",
                             channels=[dict(name='Dev2/ai0'),
                                       dict(name='Dev2/di0'),
                                       dict(name='Dev2/ao0'),
                                       dict(name='Dev2/do1')],
                             sample_rate=100,
                             labels=['t(s)', 'ai0', 'di0'],
                             cmd_labels=['cmd', 'cmd2'])
  crappy.link(gen, io)
  crappy.link(gen, io, modifier=partial(change_name, prev='cmd', new='cmd2'))
  graph = crappy.blocks.Grapher(('t(s)', 'di0'), ('t(s)', 'ai0'))
  crappy.link(io, graph, modifier=intify)
  crappy.start()
