#coding: utf-8

from __future__ import print_function, division

#ex: dwSetParam with auto error check
# Add pyspcm.py and py_header folder to crappy/inout/ (see spectrum examples)
from pyspcm import * # Temporary, own wrapper in progress
#from ..tool import pyspcm # Todo: define my own wrapper,
import numpy as np
from time import time
from .inout import InOut

class SpectrumError(Exception):
  pass

class Spectrum(InOut):
  def __init__(self,**kwargs):
    InOut.__init__(self)
    for arg,default in [('device', b'/dev/spcm0'),
                        ('channels',[0]),
                        ('ranges',[1000]),
                        ('freq',100000),
                        ('buff_size',2**22),  # 4 MB
                        ('notify_size',2**16) # 64 kB
                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    if kwargs:
      raise AttributeError("Invalid arg for Spectrum: "+str(kwargs))
    self.nchan = len(self.channels)
    print("[Spectrum] Will send {} chunks of {} kB per second ({} kB/s)".format(
      2*self.freq*self.nchan/self.notify_size,
      self.notify_size/1024,
      self.freq*self.nchan/512))
    self.bs = self.notify_size//(2*self.nchan)

  def check(self,code=-1):
    if code == 0:
      return
    print("Error: return code=",code)
    szErrorTextBuffer = create_string_buffer(ERRORTEXTLEN)
    spcm_dwGetErrorInfo_i32(self.h, None, None, szErrorTextBuffer)
    sys.stdout.write("{0}\n".format(szErrorTextBuffer.value))
    spcm_vClose(self.h)
    raise SpectrumError(szErrorTextBuffer.value)

  def open(self):
    self.h = spcm_hOpen(self.device)
    if not self.h:
      raise IOError("Could not open "+str(self.device))
    self.check(spcm_dwSetParam_i32(self.h, SPC_CHENABLE,
        sum([2**c for c in self.channels])))
    spcm_dwSetParam_i32(self.h, SPC_CARDMODE, SPC_REC_FIFO_SINGLE)
    spcm_dwSetParam_i32(self.h, SPC_TIMEOUT, 5000)
    spcm_dwSetParam_i32(self.h, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
    spcm_dwSetParam_i32(self.h, SPC_TRIG_ANDMASK, 0)
    spcm_dwSetParam_i32(self.h, SPC_CLOCKMODE, SPC_CM_INTPLL)
    for i,chan in enumerate(self.channels):
      spcm_dwSetParam_i32(self.h, SPC_AMP0+100*chan, int32(self.ranges[i]))

    spcm_dwSetParam_i64(self.h, SPC_SAMPLERATE, self.freq)
    spcm_dwSetParam_i32(self.h, SPC_CLOCKOUT, 0)
    realFreq = uint64(0)
    spcm_dwGetParam_i64(self.h, SPC_SAMPLERATE, byref(realFreq))
    self.dt = 1/realFreq.value

    self.buff = create_string_buffer(self.buff_size) # Allocating the buffer

    spcm_dwDefTransfer_i64(self.h,
                           SPCM_BUF_DATA,
                           SPCM_DIR_CARDTOPC,
                           int32(self.notify_size),
                           self.buff,
                           uint64(0),
                           uint64(self.buff_size))

    self.status = int32()
    self.avail = int32()
    self.pcpos = int32()

  def close(self):
    if hasattr(self,"h") and self.h:
      spcm_vClose(self.h)

  def start_stream(self):
    self.check(spcm_dwSetParam_i32(self.h, SPC_M2CMD, M2CMD_CARD_START
                                          | M2CMD_CARD_ENABLETRIGGER
                                          | M2CMD_DATA_STARTDMA))
    self.t0 = time()
    self.n = 0

  def get_stream(self):
    start = self.t0 + self.dt*self.n
    t = np.arange(start,start+(self.bs-1)*self.dt,self.dt)
    self.check(spcm_dwSetParam_i32(self.h,SPC_M2CMD,M2CMD_DATA_WAITDMA))
    spcm_dwGetParam_i32(self.h, SPC_M2STATUS, byref(self.status))
    spcm_dwGetParam_i32(self.h, SPC_DATA_AVAIL_USER_LEN, byref(self.avail))
    spcm_dwGetParam_i32(self.h, SPC_DATA_AVAIL_USER_POS, byref(self.pcpos))
    #print("We have {} bytes available, start at position {}".format())
    a = np.frombuffer(self.buff,dtype=np.int16,
                          count=self.notify_size//2,
                          offset=self.pcpos.value)\
        .reshape(self.notify_size//(2*self.nchan),self.nchan)
    # To return mV as floats (More CPU and memory!)
    # =======================
    #r = np.empty(a.shape)
    #for i in range(len(self.channels)):
    #  r[:,i] = a[:,i]*self.ranges[i]/32000
    # =======================
    # To return ints
    # =======================
    r = a.copy()
    # =======================
    del a
    spcm_dwSetParam_i32(self.h, SPC_DATA_AVAIL_CARD_LEN,  int32(self.notify_size))
    self.n += self.bs
    return [t]+[r[:,i] for i in range(len(self.channels))]
    #total += notify_size
