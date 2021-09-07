# coding: utf-8

"""More documentation coming soon !"""

from ctypes import *
from .._global import OptionalModule

try:
  c = cdll.LoadLibrary("/usr/lib/libcomedi.so")
  is_installed = True
except OSError:
  c = OptionalModule('libcomedi.so')
  is_installed = False


class comedi_t(Structure):
  pass


class comedi_range(Structure):
  pass


if is_installed:
  AREF_GROUND = 0
  lsampl_t = c_uint
  data = lsampl_t()  # To store data for data_read before returning it

  comedi_open = c.comedi_open
  comedi_open.restype = POINTER(comedi_t)  # Device handle
  comedi_open.argtypes = [c_char_p]  # Device path

  comedi_close = c.comedi_close
  comedi_close.restype = c_int  # Error code: 0 on success else -1
  comedi_close.argtypes = [POINTER(comedi_t)]  # device handle

  comedi_to_phys = c.comedi_to_phys
  comedi_to_phys.restype = c_double  # Physical value
  comedi_to_phys.argtypes = [lsampl_t, POINTER(comedi_range), lsampl_t]
  # Comedi value, range, maxdata

  comedi_from_phys = c.comedi_from_phys
  comedi_from_phys.restype = lsampl_t  # Comedi value
  comedi_from_phys.argtypes = [c_double, POINTER(comedi_range), lsampl_t]
  # Physical value, range, maxdata

  comedi_get_maxdata = c.comedi_get_maxdata
  comedi_get_maxdata.restype = lsampl_t  # Maxdata
  comedi_get_maxdata.argtypes = [POINTER(comedi_t), c_uint, c_uint]
  # handle, subdevice, channel

  comedi_get_range = c.comedi_get_range
  comedi_get_range.restype = POINTER(comedi_range)  # Range
  comedi_get_range.argtypes = [POINTER(comedi_t), c_uint, c_uint, c_uint]
  # handle, subdevice, channel, range_num

  c.comedi_data_read.restype = c_int  # Error code (1 if fine else -1)
  c.comedi_data_read.argtypes = [POINTER(comedi_t),
                                 c_uint, c_uint, c_uint,
                                 c_uint, POINTER(lsampl_t)]
  # Handle, subdevice, channel, range_num, aref, data pointer


  def comedi_data_read(*args):
    assert c.comedi_data_read(*(args+(byref(data),))) == 1, "Data read failed!"
    return data

  comedi_data_write = c.comedi_data_write
  comedi_data_write.restype = c_int  # Error code (1 if fine else -1)
  comedi_data_write.argtypes = [POINTER(comedi_t), c_uint, c_uint, c_uint,
                                c_uint, lsampl_t]
  # Handle, subdevice, channel, range_num, aref, data


else:
  AREF_GROUND = OptionalModule('libcomedi.so')
  lsampl_t = OptionalModule('libcomedi.so')
  data = OptionalModule('libcomedi.so')

  comedi_open = OptionalModule('libcomedi.so')

  comedi_close = OptionalModule('libcomedi.so')

  comedi_to_phys = OptionalModule('libcomedi.so')

  comedi_from_phys = OptionalModule('libcomedi.so')

  comedi_get_maxdata = OptionalModule('libcomedi.so')

  comedi_get_range = OptionalModule('libcomedi.so')

  comedi_data_read = OptionalModule('libcomedi.so')

  comedi_data_write = OptionalModule('libcomedi.so')
