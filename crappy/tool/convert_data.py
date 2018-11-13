#coding: utf-8
'''
Helpful functions for converting register data to numbers and vice versa.
Useful when modbus communication is initated.
Originially got from Labjack.com:
https://labjack.com/support/software/examples/modbus/python
'''
import struct

'''
Converting numbers to 16-bit data arrays
'''


def accept(mini,maxi):
  def real_accept(f):
    def wrapper(num):
      assert mini<=num<=maxi,"Out of range!"
      return f(num)
    return wrapper
  return real_accept


@accept(0,2**16-1)
def uint16_to_data(num):
  return num & 0xFFFF,


@accept(-2**15,2**15-1)
def int16_to_data(num):
  return num & 0xFFFF,


@accept(0,2**32-1)
def uint32_to_data(num):
  return (num >>16 & 0xFFFF,num & 0xFFFF)


@accept(-2**31,2**31-1)
def int32_to_data(num):
  return (num >>16 & 0xFFFF,num & 0xFFFF)


def float32_to_data(num):
  return struct.unpack("=HH",struct.pack("=f",num))[::-1]


'''
Converting data arrays to numbers
'''


def data_to_uint16(data):
  assert len(data) == 1,"uint16 expects 1 register"
  return data[0]


def data_to_uint32(data):
  assert len(data) == 2,"uint32 expects 2 registers"
  return (data[0] << 16) + data[1]


def data_to_int32(data):
  assert len(data) == 2,"int32 expects 2 registers"
  return struct.unpack("=i", struct.pack("=HH", *data[::-1]))[0]


def data_to_int16(data):
  assert len(data) == 1,"int16 expects 1 register"
  return struct.unpack("=h", struct.pack("=H", data[0]))[0]


def data_to_float32(data):
  assert len(data) == 2,"float32 expects 2 registers"
  return struct.unpack("=f", struct.pack("=HH", *data[::-1]))[0]
