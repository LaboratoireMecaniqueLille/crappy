# coding: utf-8

from time import time

from .inout import InOut
from .._global import OptionalModule
try:
  from ..tool import comedi_bind as c
except OSError:  # Will be raised if unable to locate the .so file
  c = OptionalModule("comedi_bind", """Could not import comedi_lib, make sure 
  libcomedi.so is installed""")


class Comedi(InOut):
  """
  Comedi object, for IO with cards using comedi driver.

  Note:
    The channel-specific args can be given as a list to set it for each channel
    or as a single value to apply it to every channel.

  Args:
    - device (str, default: "/dev/comedi0"): The address of the device.
    - subdevice (int, default: 0): The id of the subdevice.
    - channels (list, default: [0]): The list of the input channels.
    - range_num (list/int, default: 0): The range to use on each channel.
    - gain (list/float, default: 1): The return value for each chan will be
      multiplied by the gain.
    - offset (list/float, default: 0): The offset will be added to the return
      value for each chan.
    - make_zero (list/bool, default: True): If True, the value read at the
      beginning will be removed to the offset to take it as a reference.
    - channels (list, default: []): The list of the output channels.
    - out_range_num (list/int, default: 0): The range to use on each output.
    - out_gain (list/float, default: 1): The output value for each chan will be
      multiplied by the gain.
    - out_offset (list/float, default: 0): The offset will be added to the
      output value for each chan.
    - out_subdevice (int, default: 1): The id of the output subdevice.

  """

  def __init__(self, **kwargs):
    InOut.__init__(self)
    self.default = {'device': b'/dev/comedi0',
                    'subdevice': 0,
                    'channels': [0],
                    'range_num': 0,
                    'gain': 1,
                    'offset': 0,
                    'make_zero': True,
                    'out_channels': [],
                    'out_range_num': 0,
                    'out_gain': 1,
                    'out_offset': 0,
                    'out_subdevice': 1
                    }
    self.kwargs = self.default
    # For consistency with out_*, in_xxx is equivalent to xxx:
    for arg in kwargs:
      if arg.startswith('in_'):
        kwargs[arg[3:]] = kwargs[arg]
        del kwargs[arg]
    self.kwargs.update(kwargs)
    self.device_name = self.kwargs['device']
    if isinstance(self.device_name, str):
      self.device_name = bytes(self.device_name, 'utf-8')
      # We need to give this to a c function, so convert it to bytes
    self.subdevice = self.kwargs['subdevice']
    self.out_subdevice = self.kwargs['out_subdevice']
    if not isinstance(self.kwargs['channels'], list):
      self.kwargs['channels'] = [self.kwargs['channels']]
    self.channels = []
    self.channels_dict = {}
    # Turning in channels kwargs into a list of dict with each channel setting
    for i, chan in enumerate(self.kwargs['channels']):
      d = {'num': chan}
      for s in ['range_num', 'gain', 'offset', 'make_zero']:
        if isinstance(self.kwargs[s], list):
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
    if not isinstance(self.kwargs['out_channels'], list):
      self.kwargs['out_channels'] = [self.kwargs['out_channels']]
    self.out_channels = []
    for i, chan in enumerate(self.kwargs['out_channels']):
      d = {'num': chan}
      for s in ['out_range_num', 'out_gain', 'out_offset']:
        if isinstance(self.kwargs[s], list):
          try:
            d[s[4:]] = self.kwargs[s][i]
          except IndexError:
            print("Lists length differ in Comedi constructor!")
            raise
        else:
          d[s[4:]] = self.kwargs[s]
      self.out_channels.append(d)

  def open(self):
    """
    Starts communication with the device, must be called before any
    set_cmd or get_data.

    Note:
      It reads channel properties from the device, those will be used
      in data_read/data_write.

    """
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
    if any([i['make_zero'] for i in self.channels]):
      off = self.eval_offset()
      for i, chan in enumerate(self.channels):
        if chan['make_zero']:
          if off[i] != off[i]:  # True if off[i] is a nan
            print("WARNING: could not measure offset on channel", chan['num'])
          else:
            chan['offset'] += off[i]

  def set_cmd(self, *cmd):
    """
    To set the value of the outputs (when specified).
    Takes as many argument as opened output channels.
    """
    assert len(cmd) == len(self.out_channels), \
      "set_cmd takes {} args, but got {}".format(
        len(self.out_channels), len(cmd))
    for val, chan in zip(cmd, self.out_channels):
      val = val * chan['gain'] + chan['offset']
      out_a = c.comedi_from_phys(val, chan['range_ds'], chan['maxdata'])
      c.comedi_data_write(self.device, self.out_subdevice, chan['num'],
                          chan['range_num'], c.AREF_GROUND, out_a)

  def get_data(self, channel="all"):
    """
    To read the value on input_channels.

    Note:
      If channel is specified, it will only read and return these channels.

      'all' (default) will read all opened channels.

    """
    if channel == 'all':
      to_read = self.channels
    else:
      if not isinstance(channel, list):
        channel = [channel]
      to_read = [self.channels[self.channels_dict[i]] for i in channel]

    data = [time()]
    for chan in to_read:
      data_read = c.comedi_data_read(self.device,
                                     self.subdevice,
                                     chan['num'],
                                     chan['range_num'],
                                     c.AREF_GROUND)

      val = c.comedi_to_phys(data_read, chan['range_ds'], chan['maxdata'])
      data.append(val * chan['gain'] + chan['offset'])
    return data

  def close(self):
    ret = c.comedi_close(self.device)
    if ret != 0:
      print('Comedi.close failed')
