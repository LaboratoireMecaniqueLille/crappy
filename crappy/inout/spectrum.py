# coding: utf-8

import numpy as np
from time import time
from .inout import InOut
from .._global import OptionalModule
try:
  from ..tool import pyspcm as spc  # Wrapper for the spcm library
except OSError:
  spc = OptionalModule("pyspcm", "Please install the Spectrum library")


class SpectrumError(Exception):
  pass


class Spectrum(InOut):
  """
  Acquire data from a Spectrum device.

  Args:
    - device (str, default: "/dev/spcm0"): The address of the device to use.
    - channels (list, default: [0]): The channels to open. See doc for the
      allowed combinations!
    - ranges (list, default: [10000]): The ranges of the channels in mV.
    - samplerate (int, default: 100000): The samplerate for all channels in Hz.
    - buff_size (int, default: 2**16 (64MB)): The size of the memory
      allocated as a rolling buffer to copy the data from the card
    - notify_size (int, default: 2**16 (64kB)): The size of the chunks of
      data to copy from the card.
    - split_chan (bool, default: False): If False, it will return a single
      2D array, else each chan will be a 1D array.

  """

  def __init__(self,
               device=b'/dev/spcm0',
               channels=None,
               ranges=None,
               samplerate=100000,
               buff_size=2**26,
               notify_size=2**16,
               split_chan=False):
    InOut.__init__(self)
    self.device = device
    self.channels = [0] if channels is None else channels
    self.ranges = [10000] if ranges is None else ranges
    self.samplerate = samplerate
    self.buff_size = buff_size
    self.notify_size = notify_size
    self.split_chan = split_chan
    self.nchan = len(self.channels)
    print("[Spectrum] Will send {} chunks of {} kB per second ({} kB/s)".
          format(2 * self.samplerate * self.nchan / self.notify_size,
             self.notify_size / 1024,
             self.samplerate * self.nchan / 512))
    self.bs = self.notify_size // (2 * self.nchan)

  def open(self):
    self.h = spc.hOpen(self.device)
    if not self.h:
      raise IOError("Could not open " + str(self.device))
    spc.dw_set_param(self.h, spc.SPC_CHENABLE,
                     sum([2 ** c for c in self.channels]))
    spc.dw_set_param(self.h, spc.SPC_CARDMODE, spc.SPC_REC_FIFO_SINGLE)
    spc.dw_set_param(self.h, spc.SPC_TIMEOUT, 5000)
    spc.dw_set_param(self.h, spc.SPC_TRIG_ORMASK, spc.SPC_TMASK_SOFTWARE)
    spc.dw_set_param(self.h, spc.SPC_TRIG_ANDMASK, 0)
    spc.dw_set_param(self.h, spc.SPC_CLOCKMODE, spc.SPC_CM_INTPLL)
    for i, chan in enumerate(self.channels):
      spc.dw_set_param(self.h, spc.SPC_AMP0 + 100 * chan, self.ranges[i])

    spc.dw_set_param(self.h, spc.SPC_SAMPLERATE, self.samplerate)
    spc.dw_set_param(self.h, spc.SPC_CLOCKOUT, 0)
    real_samplerate = spc.dw_get_param(self.h, spc.SPC_SAMPLERATE)
    self.dt = 1 / real_samplerate

    self.buff = spc.new_buffer(self.buff_size)  # Allocating the buffer

    spc.dw_def_transfer(self.h,  # Handle
                      spc.SPCM_BUF_DATA,  # Buff type
                      spc.SPCM_DIR_CARDTOPC,  # Direction
                      self.notify_size,  # Notify every x byte
                      self.buff,  # buffer
                      0,  # Offset
                      self.buff_size)  # Buffer size

  def close(self):
    if hasattr(self, "h") and self.h:
      spc.vClose(self.h)

  def start_stream(self):
    spc.dw_set_param(self.h, spc.SPC_M2CMD, spc.M2CMD_CARD_START |
                                            spc.M2CMD_CARD_ENABLETRIGGER |
                                            spc.M2CMD_DATA_STARTDMA)
    self.t0 = time()
    self.n = 0

  def get_stream(self):
    start = self.t0 + self.dt * self.n
    t = np.arange(start, start + (self.bs - 1) * self.dt, self.dt)
    spc.dw_set_param(self.h, spc.SPC_M2CMD, spc.M2CMD_DATA_WAITDMA)
    # self.status = spc.dw_get_param(self.h, spc.SPC_M2STATUS)
    # self.avail = spc.dw_get_param(self.h, spc.SPC_DATA_AVAIL_USER_LEN)
    self.pcpos = spc.dw_get_param(self.h, spc.SPC_DATA_AVAIL_USER_POS)
    a = np.frombuffer(self.buff, dtype=np.int16,
                          count=self.notify_size//2,
                          offset=self.pcpos)\
        .reshape(self.notify_size//(2*self.nchan), self.nchan)
    # To return mV as floats (More CPU and memory!)
    # =======================
    # r = np.empty(a.shape)
    # for i in range(len(self.channels)):
    #  r[:,i] = a[:,i] * self.ranges[i] / 32000
    # =======================
    # To return ints
    # =======================
    r = a.copy()
    # =======================
    del a
    spc.dw_set_param(self.h, spc.SPC_DATA_AVAIL_CARD_LEN, self.notify_size)
    self.n += self.bs
    if self.split_chan:
      return [t] + [r[:, i] for i in range(len(self.channels))]
    else:
      return [t, r]
    # total += notify_size

  def stop_stream(self):
    spc.dw_set_param(self.h, spc.SPC_M2CMD, spc.M2CMD_CARD_STOP |
                                          spc.M2CMD_DATA_STOPDMA)
