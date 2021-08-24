# coding: utf-8

"""
Example illustrating the use of Spectrum DAQ cards and high-freq saving.

The data is saved using the hdf format.

Required hardware:
  - Spectrum acquisition board
"""

import crappy

if __name__ == "__main__":
  channels = list(range(16))  # Every channel (from 0 to 15)
  ranges = [10000] * len(channels)  # -10/+10V (in mV)
  # This will NOT apply the gain to the stream, only save a key in the h5
  gains = [1] * len(channels)
  save_file = "./out.h5"

  chan_names = ['ch' + str(i) for i in channels]

  spectrum = crappy.blocks.IOBlock('spectrum', ranges=ranges,
                                   channels=channels,
                                   streamer=True,
                                   labels=['t(s)', 'stream'])

  graph = crappy.blocks.Grapher(*[('t(s)', i) for i in chan_names])

  if save_file:
    hrec = crappy.blocks.Hdf_recorder("./out.h5",
                                      metadata={
                                        'channels': channels,
                                        'ranges': ranges,
                                        'freq': 100000,
                                        'factor': [r * g / 32000000 for r, g in
                                                   zip(ranges, gains)]})
    crappy.link(spectrum, hrec)

  # The Demux modifier unpacks the stream data so normal blocks can process it
  crappy.link(spectrum, graph,
              modifier=crappy.modifier.Demux(chan_names, mean=False))
  # Note that the majority of the data is DROPPED with this modifier, it was
  # meant to plot the data in real time at a much lower rate
  # However, the hdf file contains the complete raw data

  crappy.start()
