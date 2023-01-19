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

  Not doing to results in an error in the grapher.
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

  # A Generator outputting a square signal oscillating between 0 and 1
  gen = crappy.blocks.Generator([dict(type='Cyclic', value1=0, value2=1,
                                      condition1="delay=1",
                                      condition2="delay=1")], repeat=True)

  # The Block communicating with the DAQ board
  io = crappy.blocks.IOBlock("NIDAQmx",
                             device="Dev2",
                             channels=[dict(name='ai0'), dict(name='di0'),
                                       dict(name='ao0'), dict(name='do1')],
                             samplerate=100,
                             labels=['t(s)', 'ai0', 'di0'],
                             cmd_labels=['cmd', 'cmd2'])
  crappy.link(gen, io)
  crappy.link(gen, io, modifier=partial(change_name, prev='cmd', new='cmd2'))

  # The Block displaying the acquired data in real-time
  graph = crappy.blocks.Grapher(('t(s)', 'di0'), ('t(s)', 'ai0'))
  crappy.link(io, graph, modifier=intify)

  # Starting the test
  crappy.start()
