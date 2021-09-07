# coding: utf-8

"""More documentation coming soon !"""

from ctypes import *
from sys import platform
from .._global import OptionalModule

# ======= Constants definition ======

ERRORTEXTLEN = 200
SPC_CHENABLE = 11000
SPC_CARDMODE = 9500
SPC_REC_STD_SINGLE = 0x00000001
SPC_REC_FIFO_SINGLE = 0x00000010
SPC_TIMEOUT = 295130
SPC_TRIG_ORMASK = 40410
SPC_TRIG_ANDMASK = 40430
SPC_TMASK_SOFTWARE = 0x00000001
SPC_CLOCKMODE = 20200
SPC_CM_INTPLL = 0x00000001
SPC_AMP0 = 30010
SPC_SAMPLERATE = 20000
SPC_CLOCKOUT = 20110
SPCM_DIR_PCTOCARD = 0
SPCM_DIR_CARDTOPC = 1
SPCM_BUF_DATA = 1000
SPCM_BUF_ABA = 2000
SPCM_BUF_TIMESTAMP = 3000
SPC_M2CMD = 100
M2CMD_CARD_RESET = 0x00000001
M2CMD_CARD_START = 0x00000004
M2CMD_CARD_ENABLETRIGGER = 0x00000008
M2CMD_CARD_DISABLETRIGGER = 0x00000020
M2CMD_CARD_STOP = 0x00000040
M2CMD_CARD_FLUSHFIFO = 0x00000080
M2CMD_CARD_INVALIDATEDATA = 0x00000100
M2CMD_ALL_STOP = 0x00440060
M2CMD_CARD_WAITPREFULL = 0x00001000
M2CMD_CARD_WAITTRIGGER = 0x00002000
M2CMD_CARD_WAITREADY = 0x00004000
M2CMD_DATA_STARTDMA = 0x00010000
M2CMD_DATA_WAITDMA = 0x00020000
M2CMD_DATA_STOPDMA = 0x00040000
M2CMD_DATA_POLL = 0x00080000
SPC_M2STATUS = 110
SPC_DATA_AVAIL_USER_LEN = 200
SPC_DATA_AVAIL_USER_POS = 201
SPC_DATA_AVAIL_CARD_LEN = 202

# ====== Library binding ========

double_reg = [SPC_SAMPLERATE]  # Place here the registers holding 64 bits data
new_buffer = create_string_buffer  # To allow to create a buffer without ctypes

if "linux" in platform.lower():
  msg = "libspcm_linux.so"
  try:
    mod = cdll.LoadLibrary("libspcm_linux.so")
    is_installed = True
  except OSError:
    mod = OptionalModule(msg)
    is_installed = False
elif "darwin" in platform.lower():
  msg = "libc.so.6"
  try:
    mod = cdll.LoadLibrary("libc.so.6")
    is_installed = True
  except OSError:
    mod = OptionalModule(msg)
    is_installed = False
else:
  msg = "spcm_win64.dll"
  try:
    mod = windll.LoadLibrary("C:\\Windows\\system32\\spcm_win64.dll")
    is_installed = True
  except OSError:
    mod = OptionalModule(msg)
    is_installed = False


class SpectrumError(Exception):
  pass


if is_installed:
  dwGetErrorInfo_i32 = mod.spcm_dwGetErrorInfo_i32
  dwGetErrorInfo_i32.argtype = [
      c_void_p, POINTER(c_uint32), POINTER(c_int32), c_char_p]
  dwGetErrorInfo_i32.restype = c_uint32


  def check(h, code):
    if code == 0:
      return
    print("Error: return code=", code)
    sz_error_text_buffer = create_string_buffer(ERRORTEXTLEN)
    dwGetErrorInfo_i32(h, None, None, sz_error_text_buffer)
    print(sz_error_text_buffer.value)
    vClose(h)
    raise SpectrumError(sz_error_text_buffer.value)


  hOpen = mod.spcm_hOpen
  hOpen.argtype = [c_char_p]
  hOpen.restype = c_void_p

  vClose = mod.spcm_vClose
  vClose.argtype = [c_char_p]
  vClose.restype = None

  my_i64 = c_int64()  # C ints to read args when calling getparam
  my_i32 = c_int32()

  mod.spcm_dwGetParam_i32.argtype = [c_void_p, c_int32, POINTER(c_int32)]
  mod.spcm_dwGetParam_i32.restype = c_uint32

  mod.spcm_dwGetParam_i64.argtype = [c_void_p, c_int32, POINTER(c_int64)]
  mod.spcm_dwGetParam_i64.restype = c_uint32


  def dw_get_param(h, reg):
    if reg in double_reg:
      check(h, mod.spcm_dwGetParam_i64(h, reg, byref(my_i64)))
      return my_i64.value
    else:
      check(h, mod.spcm_dwGetParam_i32(h, reg, byref(my_i32)))
      return my_i32.value


  mod.spcm_dwSetParam_i32.argtype = [c_void_p, c_int32, c_int32]
  mod.spcm_dwSetParam_i32.restype = c_uint32

  mod.spcm_dwSetParam_i64.argtype = [c_void_p, c_int32, c_int64]
  mod.spcm_dwSetParam_i64.restype = c_uint32


  def dw_set_param(h, reg, val):
    if reg in double_reg:
      check(h, mod.spcm_dwSetParam_i64(h, reg, val))
    else:
      check(h, mod.spcm_dwSetParam_i32(h, reg, val))


  mod.spcm_dwDefTransfer_i64.argtype = [c_void_p, c_uint32, c_uint32, c_uint32,
                                        c_void_p, c_uint64, c_uint64]
  mod.spcm_dwDefTransfer_i64.restype = c_uint32


  def dw_def_transfer(h, buff_type, direction, notify_size, buff, offset,
                      buff_size):
    check(h, mod.spcm_dwDefTransfer_i64(h, buff_type, direction, notify_size,
                                        buff, c_uint64(offset),
                                        c_uint64(buff_size)))

else:
  dwGetErrorInfo_i32 = OptionalModule(msg)

  hOpen = OptionalModule(msg)

  vClose = OptionalModule(msg)

  my_i64 = OptionalModule(msg)
  my_i32 = OptionalModule(msg)

  dw_get_param = OptionalModule(msg)

  dw_set_param = OptionalModule(msg)

  dw_def_transfer = OptionalModule(msg)
