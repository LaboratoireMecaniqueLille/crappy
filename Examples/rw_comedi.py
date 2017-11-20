#coding: utf-8

import crappy

"""
This example shows how to use multiple output channels object while acquiring
with an inout object
We use 2 generators to make two independant signals, each of them having
a different cmd label
We simply need to specify these labels in IOBlock and it will automatically
parse the inputs to get the latest value of the when updating the output.

Note: this exemple uses a Labjack but can run on other devices (you may
have to ajust the range for the Comedi)
"""

sg1 = crappy.blocks.Generator([{'type':'sine','freq':.5,'amplitude':.4,
  'offset':.2,'condition':'delay=1000'}])

sg2 = crappy.blocks.Generator([{'type':'cyclic_ramp','speed1':.2,'speed2':-.2,
  'condition1':'cmd2>.4','condition2':'cmd2<0','cycles':0}],cmd_label='cmd2')

io = crappy.blocks.IOBlock('Comedi',labels=['t(s)','c0','c1'],channels=[0,1],
    out_channels=[0,1],verbose=True,cmd_labels=['cmd','cmd2'],make_zero=False)

crappy.link(sg1,io)
crappy.link(sg2,io)

g = crappy.blocks.Grapher(('t(s)','c0'),('t(s)','c1'))

crappy.link(io,g)

crappy.start()
