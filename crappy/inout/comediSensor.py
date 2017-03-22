# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup comedisensor ComediSensor
# @{

## @file _comediSensor.py
# @brief  Sensor class for Comedi devices.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import comedi as c
import time

from inout import InOut


class ComediSensor(InOut):
  """
  Sensor class for Comedi devices.
  """

  def __init__(self, *args, **kwargs):
    """
    Convert tension value into digital values, on several channels.

    Output is (measured_value * gain) + offset.

    Args:
        device : str, default = '/dev/comedi0'
                Path to the device.
        subdevice : int, default = 0
        Subdevice 0 is the input.
        channel : int or list of int, default = 0
                The desired output channel(s).
        range_num : int, default = 0
                    See the comedi documentation for different values.
        gain : float or list of float, default = 1
                Multiplication gain for each channel. If there are multiple
                channels
                for a single gain, it will be applied to all.
        offset : float, default = 0
                Add this value for each channel. If there is multiple channels
                for a single offset, it will be applied to all.
    """
    def var_tester(var, nb_channels):
      """Used to check if the user entered correct parameters."""
      var = [var] * nb_channels if isinstance(var, (int, float)) else var
      assert isinstance(var, list) and len(var) == nb_channels, \
        str(var) + "Parameter definition Error: list is" \
        " not the same length as nb_channels."
      assert False not in [isinstance(var[i], (int, float)) for i in
                           xrange(nb_channels)], \
        str(var) + "Error: parameter should be int or float."
      return var

    super(ComediSensor, self).__init__()
    self.subdevice = kwargs.get('subdevice', 0)  # input subdevice number
    self.channels = kwargs.get('channels', [0])
    if not isinstance(self.channels, list):
      self.channels = [self.channels]
    self.nb_channels = len(self.channels)

    self.range_num = var_tester(kwargs.get('range_num', 0), self.nb_channels)
    self.gain = var_tester(kwargs.get('gain', 1), self.nb_channels)
    self.offset = var_tester(kwargs.get('offset', 0), self.nb_channels)

    self.device = kwargs.get('device', '/dev/comedi0')
    self.device = c.comedi_open(self.device)
    self.new()

  def get_data(self, channel_number=0):
    """
    Read the signal for desired channel
    """
    if channel_number == "all":
      result = []
      for channel in xrange(self.nb_channels):
        data = c.comedi_data_read(self.device,
                                  self.subdevice,
                                  self.channels[channel],
                                  self.range_num[channel],
                                  c.AREF_GROUND)

        self.position = (c.comedi_to_phys(data[1], self.range_ds[channel],
                                          self.maxdata[channel]) *
                         self.gain[channel] + self.offset[channel])

        result.append(self.position)
      t = time.time()
      return t, result

    else:
      data = c.comedi_data_read(self.device,
                                self.subdevice,
                                self.channels[channel_number],
                                self.range_num[channel_number],
                                c.AREF_GROUND)

      self.position = (c.comedi_to_phys(data[1], self.range_ds[channel_number],
                                        self.maxdata[channel_number]) *
                       self.gain[channel_number] + self.offset[
                         channel_number])
      t = time.time()
      return t, self.position

  def open(self):
    pass
  def new(self):
    """
    Gather range and maxdata for all specified channels.

    This is only called on init.
    """
    self.maxdata = [0] * self.nb_channels
    self.range_ds = [0] * self.nb_channels
    for i in xrange(self.nb_channels):
      self.maxdata[i] = c.comedi_get_maxdata(self.device,
                                             self.subdevice,
                                             self.channels[i])
      self.range_ds[i] = c.comedi_get_range(self.device,
                                            self.subdevice,
                                            self.channels[i],
                                            self.range_num[i])

  def close(self):
    """
    Close the device.
    """
    c.comedi_cancel(self.device, self.subdevice)
    ret = c.comedi_close(self.device)
    if ret != 0: raise Exception('comedi_close failed...')
