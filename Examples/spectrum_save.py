#coding: utf-8

import crappy

channels = list(range(16)) # Every channel (from 0 to 15)
ranges = [10000]*len(channels) # -10/+10V (in mV)

chan_names = ['ch'+str(i) for i in channels]

spectrum = crappy.blocks.IOBlock('spectrum',ranges=ranges,
    channels=channels,
    streamer=True,
    labels=['t(s)','stream'])

graph = crappy.blocks.Grapher(*[('t(s)',i) for i in chan_names])
hsaver = crappy.blocks.Hdf_saver("./out.h5",
    metadata={'channels':channels,'ranges':ranges,'freq':100000})
crappy.link(spectrum,hsaver)
crappy.link(spectrum,graph,
    condition=crappy.condition.Demux(chan_names,mean=False))

crappy.start()
