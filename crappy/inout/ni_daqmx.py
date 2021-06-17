# coding: utf-8

from time import time
import numpy as np

from .inout import InOut
from .._global import OptionalModule
try:
  import nidaqmx
  from nidaqmx import stream_readers, stream_writers
except (ModuleNotFoundError, ImportError):
  nidaqmx = OptionalModule("nidaqmx")
  stream_readers = stream_writers = nidaqmx


class Nidaqmx(InOut):
  """Opens National Instrument devices using the NiDAQmx driver (Windows only).

  This class can open `ai`, `ao` and `dio` channels on NI devices. It supports
  continuous streaming on `ai` channels only. Streaming is not supported with
  `ai` channels of different types.

  It uses the :mod:`nidaqmx` python module by NI.
  """

  def __init__(self,
               channels=None,
               samplerate=100,
               nsamples=None):
    """Builds the different channels lists.

    Args:
      channels (:obj:`list`, optional): A :obj:`list` of :obj:`dict` describing
        all the channels to open. See the note below for details.

        Note:
          For `dio`, use `DevX/d[i/o] Y` to select `port(Y//8)/line(Y%8)` on
          `DevX`.

      samplerate (:obj:`float`, optional): If using stream mode, the samplerate
        of the stream.
      nsamples (:obj:`int`, optional): If using stream mode, the stream array
        will be returned after reading ``nsamples`` samples. Defaults to
        ``samplerate // 5``.

    Note:
      - ``channels`` keys:

        - name (:obj:`str`): The name of the channel to open. For dio, use
          `diX` for digital input on line `X` and `doX` for digital output.
        - type (:obj:`str`, default: 'voltage'): The type of channel to open.
          Ex: `'thrmcpl'`.
        - All the other args will be given as kwargs to
          `nidaqmx.Task.add_[a/d][i/o]_[type]_chan`.
    """

    InOut.__init__(self)
    self.thermocouple_type = {"B": nidaqmx.constants.ThermocoupleType.B,
                              "E": nidaqmx.constants.ThermocoupleType.E,
                              "J": nidaqmx.constants.ThermocoupleType.J,
                              "K": nidaqmx.constants.ThermocoupleType.K,
                              "N": nidaqmx.constants.ThermocoupleType.N,
                              "R": nidaqmx.constants.ThermocoupleType.R,
                              "S": nidaqmx.constants.ThermocoupleType.S,
                              "T": nidaqmx.constants.ThermocoupleType.T,
                              }
    self.units = {"C": nidaqmx.constants.TemperatureUnits.DEG_C,
                  "F": nidaqmx.constants.TemperatureUnits.DEG_F,
                  "R": nidaqmx.constants.TemperatureUnits.DEG_R,
                  "K": nidaqmx.constants.TemperatureUnits.K,
                  # To be completed...
                  }

    self.channels = [{'name': 'Dev1/ai0'}] if channels is None else channels
    self.samplerate = samplerate
    self.nsamples = nsamples

    if self.nsamples is None:
      self.nsamples = max(1, int(self.samplerate / 5))
    self.streaming = False
    self.ao_channels = []
    self.ai_channels = {}
    self.di_channels = []
    self.do_channels = []
    for c in self.channels:
      c['type'] = c.get('type', 'voltage').lower()
      if c['name'].split('/')[-1].startswith('ai'):
        if c['type'] in self.ai_channels:
          self.ai_channels[c['type']].append(c)
        else:
          self.ai_channels[c['type']] = [c]
      elif c['name'].split('/')[-1].startswith('ao'):
        self.ao_channels.append(c)
      elif c['name'].split('/')[-1].startswith('di'):
        i = int(c['name'].split('/')[-1][2:])
        c['name'] = "/".join(
            [c['name'].split("/")[0], 'port%d' % (i // 8), 'line%d' % (i % 8)])
        self.di_channels.append(c)
      elif c['name'].split('/')[-1].startswith('do'):
        i = int(c['name'].split('/')[-1][2:])
        c['name'] = "/".join(
            [c['name'].split("/")[0], 'port%d' % (i // 8), 'line%d' % (i % 8)])
        self.do_channels.append(c)
      else:
        raise AttributeError("Unknown channel in nidaqmx" + str(c))
    self.ai_chan_list = sum(self.ai_channels.values(), [])

  def open(self):
    # AI
    self.t_in = {}

    for c in self.ai_chan_list:
      kwargs = dict(c)
      kwargs.pop("name", None)
      kwargs.pop("type", None)
      if c['type'] == 'voltage':
        kwargs['max_val'] = kwargs.get('max_val', 5)
        kwargs['min_val'] = kwargs.get('min_val', 0)
      for k in kwargs:
        if isinstance(kwargs[k], str):
          if k == "thermocouple_type":
            kwargs[k] = self.thermocouple_type[kwargs[k]]
          elif k == "units":
            kwargs[k] = self.units[kwargs[k]]
      if not c['type'] in self.t_in:
        self.t_in[c['type']] = nidaqmx.Task()
      try:
        getattr(self.t_in[c['type']].ai_channels,
                "add_ai_%s_chan" % c['type'])(c['name'], **kwargs)
      except Exception:
        print("Invalid channel settings in nidaqmx:" + str(c))
        raise
    self.stream_in = {}
    for chan_type, t in self.t_in.items():
      self.stream_in[chan_type] = \
          stream_readers.AnalogMultiChannelReader(t.in_stream)
    # AO
    if self.ao_channels:
      self.t_out = nidaqmx.Task()
      self.stream_out = stream_writers.AnalogMultiChannelWriter(
          self.t_out.out_stream, auto_start=True)

    for c in self.ao_channels:
      kwargs = dict(c)
      kwargs.pop("name", None)
      kwargs.pop("type", None)
      kwargs['max_val'] = kwargs.get('max_val', 5)
      kwargs['min_val'] = kwargs.get('min_val', 0)
      self.t_out.ao_channels.add_ao_voltage_chan(c['name'], **kwargs)
    # DI
    if self.di_channels:
      self.t_di = nidaqmx.Task()
    for c in self.di_channels:
      kwargs = dict(c)
      kwargs.pop("name", None)
      kwargs.pop("type", None)
      self.t_di.di_channels.add_di_chan(c['name'], **kwargs)
    if self.di_channels:
      self.di_stream = stream_readers.DigitalMultiChannelReader(
          self.t_di.in_stream)

    # DO
    if self.do_channels:
      self.t_do = nidaqmx.Task()
    for c in self.do_channels:
      kwargs = dict(c)
      kwargs.pop("name", None)
      kwargs.pop("type", None)
      self.t_do.do_channels.add_do_chan(c['name'], **kwargs)
    if self.do_channels:
      self.do_stream = stream_writers.DigitalMultiChannelWriter(
          self.t_do.out_stream)

  def start_stream(self):
    if len(self.t_in) != 1:
      raise IOError("Stream mode can only open one type of chan!")
    for t in self.t_in.values():  # Only one loop
      t.timing.cfg_samp_clk_timing(self.samplerate,
          sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    self.streaming = True

  def stop_stream(self):
    self.streaming = False
    for t in self.t_in.values():
      t.stop()
      t.timing.cfg_samp_clk_timing(self.samplerate,
        sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

  def get_stream(self):
    if not self.streaming:
      self.start_stream()
    a = np.empty((len(self.ai_chan_list), self.nsamples))
    for chan_type, s in self.stream_in.items():  # Will only loop once
      s.read_many_sample(a, self.nsamples)
    return [time(), a]

  def close(self):
    for t in self.t_in.values():
      t.stop()
    if self.ao_channels:
      self.t_out.stop()
    if self.streaming:
      self.stop_stream()
    for t in self.t_in.values():
      t.close()
    if self.ao_channels:
      self.t_out.close()

  def get_data(self):
    a = np.empty((sum([len(i) for i in self.ai_channels.values()]),))
    i = 0
    t = time()
    for chan_type, s in self.stream_in.items():
      s.read_one_sample(a[i:i+len(self.ai_channels[chan_type])])
      i += len(self.ai_channels[chan_type])
    if self.di_channels:
      b = np.empty((len(self.di_channels), 1), dtype=np.bool)
      self.di_stream.read_one_sample_multi_line(b)
      return [t] + list(a)+list(b[:, 0])
    return [t] + list(a)

  def set_cmd(self, *v):
    if self.ao_channels:
      self.stream_out.write_one_sample(np.array(v[:len(self.ao_channels)],
        dtype=np.float64))
    if self.do_channels:
      self.do_stream.write_one_sample_multi_line(
        np.array(v[len(self.ao_channels):],
                 dtype=np.bool).reshape(len(self.do_channels), 1))
