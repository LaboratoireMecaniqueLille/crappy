#coding: utf-8

import crappy

channels = list(range(16)) # Every channel (from 0 to 15)
ranges = [10000]*len(channels) # -10/+10V (in mV)

chan_names = ['ch'+str(i) for i in channels]

# To only pick one point of data on each chunk (to plot it)
def split_pick(data):
  for i,n in enumerate(chan_names):
    data[n] = data['stream'][0,i]
  del data['stream']
  data['t(s)'] = data['t(s)'][0]
  return data

spectrum = crappy.blocks.IOBlock('spectrum',ranges=ranges,
    channels=channels,
    notify_size=2**16,
    buff_size=2**26,
    streamer=True,
    labels=['t(s)','stream'])

graph = crappy.blocks.Grapher(*[('t(s)',i) for i in chan_names])
hsaver = crappy.blocks.Hdf_saver("/home/tribo/out.h5",
    metadata={'channels':channels,'ranges':ranges,'freq':100000})
crappy.link(spectrum,hsaver)
crappy.link(spectrum,graph,condition=split_pick)

crappy.start()
