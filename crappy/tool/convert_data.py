'''
Helpful functions for converting register data to numbers and vice versa.
Useful when modbus communication is initated.
Originially got from Labjack.com:
https://labjack.com/support/software/examples/modbus/python
'''

import struct


def float2int(num):
  return struct.unpack("=i", struct.pack("=f", num))[0]


def concatData(data):
  tVal = 0
  upper = True
  for reg in data:
    if upper:
      tVal = ((reg & 0xFFFF) << 16)
      upper = False
    else:
      tVal = tVal | (reg & 0xFFFF)
      upper = True
  return tVal


'''
Converting numbers to 16-bit data arrays
'''


def uint16_to_data(num):
  return struct.unpack("=H", struct.pack("=H", num & 0xFFFF))[0]


def uint32_to_data(num):
  data = [0, 0]
  data[0] = struct.unpack("=H", struct.pack("=H", (num >> 16) & 0xffff))[0]
  data[1] = struct.unpack("=H", struct.pack("=H", num & 0xffff))[0]
  return data


def int32_to_data(num):
  data = [0, 0]
  data[0] = struct.unpack("=H", struct.pack("=H", (num >> 16) & 0xffff))[0]
  data[1] = struct.unpack("=H", struct.pack("=H", num & 0xffff))[0]
  return data


def float32_to_data(num):
  intNum = float2int(num)
  data = [0, 0]
  data[0] = (intNum >> 16) & 0xFFFF
  data[1] = intNum & 0xFFFF
  return data


'''
Converting data arrays to numbers
'''


def data_to_uint16(data):
  return data[0]


def data_to_uint32(data):
  return concatData(data)


def data_to_int32(data):
  return struct.unpack("=i", struct.pack("=I", concatData(data)))[0]


def data_to_float32(data):
  return struct.unpack("=f", struct.pack("=I", concatData(data)))[0]
