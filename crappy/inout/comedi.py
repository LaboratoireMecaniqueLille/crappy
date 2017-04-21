# coding: utf-8
from __future__ import print_function,absolute_import

from time import time

import comedi as c

from .inout import InOut

class Comedi(InOut):
  """Comedi object, for IO with cards using comedi driver"""
  def __init__(self, **kwargs):
    InOut.__init__(self)
    self.default = {'device':'/dev/comedi0',
                    'subdevice':0,
                    'channels':[0],
                    'range_num':0,
                    'gain':1,
                    'offset':0,
                    'out_channels':[],
                    'out_range_num':0,
                    'out_gain':1,
                    'out_offset':0,
                    'out_subdevice':1
                    }
    self.kwargs = self.default
    # For consistency with out_*, in_xxx is equivalent to xxx:
    for arg in kwargs:
      if arg.startswith('in_'):
        kwargs[arg[3:]] = kwargs[arg]
        del kwargs[arg]
    self.kwargs.update(kwargs)
    self.device_name = self.kwargs['device']
    self.subdevice = self.kwargs['subdevice']
    self.out_subdevice = self.kwargs['out_subdevice']
    if not isinstance(self.kwargs['channels'],list):
      self.kwargs['channels'] = [self.kwargs['channels']]
    self.channels = []
    self.channels_dict = {}
    # Turning in channels kwargs into a list of dict with each channel setting
    for i,chan in enumerate(self.kwargs['channels']):
      d = {'num':chan}
      for s in ['range_num','gain','offset']:
        if isinstance(self.kwargs[s],list):
          try:
            d[s] = self.kwargs[s][i]
          except IndexError:
            print("Lists length differ in Comedi constructor!")
            raise
        else:
          d[s] = self.kwargs[s]
      # To quickly get the channel settings with its number (for get_data)
      self.channels_dict[chan] = len(self.channels)
      self.channels.append(d)

    # Same as above, but for out_channels
    if not isinstance(self.kwargs['out_channels'],list):
      self.kwargs['out_channels'] = [self.kwargs['out_channels']]
    self.out_channels = []
    for i,chan in enumerate(self.kwargs['out_channels']):
      d = {'num':chan}
      for s in ['out_range_num','out_gain','out_offset']:
        if isinstance(self.kwargs[s],list):
          try:
            d[s[4:]] = self.kwargs[s][i]
          except IndexError:
            print("Lists length differ in Comedi constructor!")
            raise
        else:
          d[s[4:]] = self.kwargs[s]
      self.out_channels.append(d)


  def open(self):
    """Starts commmunication with the device, must be called before any
    set_cmd or get_data"""
    self.device = c.comedi_open(self.device_name)
    for chan in self.channels:
      chan['maxdata'] = c.comedi_get_maxdata(self.device, self.subdevice,
                                          chan['num'])
      chan['range_ds'] = c.comedi_get_range(self.device, self.subdevice,
                                          chan['num'], chan['range_num'])
    for chan in self.out_channels:
      chan['maxdata'] = c.comedi_get_maxdata(self.device, self.out_subdevice,
                                          chan['num'])
      chan['range_ds'] = c.comedi_get_range(self.device, self.out_subdevice,
                                          chan['num'], chan['range_num'])
      c.comedi_dio_config(self.device, 2, chan['num'], 1)
      c.comedi_dio_write(self.device, 2, chan['num'], 1)


  def set_cmd(self, *cmd):
    """To set the value of the outputs (when specified)
    Takes as many argument as opened output channels"""
    assert len(cmd) == len(self.out_channels),\
        "set_cmd takes {} args, but got {}".format(
        len(self.out_channels),len(cmd))
    for val,chan in zip(cmd,self.out_channels):
      val = val*chan['gain']+chan['offset']
      out_a = c.comedi_from_phys(val, chan['range_ds'], chan['maxdata'])
      c.comedi_data_write(self.device, self.out_subdevice, chan['num'],
                          chan['range_num'], c.AREF_GROUND, out_a)


  def get_data(self,channel="all"):
    """To read the value on input_channels. If channel is specified, it will
    only read and return these channels. 'all' (default) will read all opened
    channels"""
    if channel == 'all':
      to_read = self.channels
    else:
      if not isinstance(channel,list):
        channel = [channel]
      to_read = [self.channels[self.channels_dict[i]] for i in channel]

    data = [time()]
    for chan in to_read:
      data_read = c.comedi_data_read(self.device,
                                self.subdevice,
                                chan['num'],
                                chan['range_num'],
                                c.AREF_GROUND)

      val = c.comedi_to_phys(data_read[1], chan['range_ds'], chan['maxdata'])
      data.append(val*chan['gain']+chan['offset'])
    return data


  def close(self):
    c.comedi_cancel(self.device, self.subdevice)
    ret = c.comedi_close(self.device)
    if ret != 0: print('Comedi.close failed')
