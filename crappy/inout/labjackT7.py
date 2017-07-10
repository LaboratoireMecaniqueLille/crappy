# coding: utf-8

from __future__ import print_function, absolute_import, division
from labjack import ljm
from time import time

from .inout import InOut


class Labjack_t7(InOut):
  """
  Class for LabJack T7 devices.

  It can use any channel as input/output, it can be used with an IOBlock.
  The keyword argument "channels" is used to specify the channels.
  Each channel must be represented as a dict including all the parameters.
  See below for more details on the parameters of the channels.
  Args:
    device: The type of the device to open (str). Ex: 'T7'.
       Can be 'ANY' (default).
    connection: How is the Labjack connected ? (str). Ex: 'USB', 'ETHERNET',..
      Can be 'ANY' (default).
    identifier: Something to identify the Labjack (str)
      It can be a name, serial number or functionality.
      Can be 'ANY' (default).
    channels: Channels to use and their settings. It must be a list of dicts.

  Channel keys:
    name: The name of the channel according to Labjack's naming convention
      (str). Ex: 'AIN0'. This will be used to define the direction (in/out)
      and the available settings.
      - AINx: An analog input, if gain and/or offset is given, the integrated
        slope mechanism will be used with the extended features registers.
        It can also be used for thermocouples (see below). You can use any
        EF by using the 'to_write' and 'to_read' keys if necessary.
      - (T)DACx: An analog output, you can specifiy gain and/or offset.
      - (E/F/C/M IOx): Digital in/outputs. You can specify the direction for each.

    gain: A numeric value that will multiply the given value for inputs
      and outputs. Default=1

    offset: Will be added to the value. Default=0
      - For inputs: returned_value = gain*measured_value+offset
      - For outputs: set_value = gain*given_value+offset
      Where measured_value and set_values are in Volts.

    make_zero: AIN only, if True the input value will be evaluated at startup
      and the offset will be adjusted to return 0 (or the offset if any).

    direction: DIO only, if True (or 1), the port will be used as an output
      else as an input.

    resolution: int 1-8(1-12) The resolution of the acquisition, see Labjack
      documentation for more details. Default=1 (fastest)

    range: 10/1/.1/.01. The range of the acquisition (V).
      10 means -10V>+10V default=10

    thermocouple: E/J/K/R/T/S/C (char) The type of thermocouple (AIN only)
      If specified, it will use the EF to read a temperature directly from
      the thermocouples.
  """
  def __init__(self, **kwargs):
    InOut.__init__(self)
    for arg, default in [
                         ('device', 'ANY'), # Model (T7, DIGIT,...)
                         ('connection', 'ANY'), # Connection (USB,ETHERNET,...)
                         ('identifier', 'ANY'), # Identifier (serial n°, ip,..)
                         ('channels', [{'name':'AIN0'}]),
                         ]:
      if arg in kwargs:
        setattr(self, arg, kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self, arg, default)
    assert len(kwargs) == 0, "Labjack_T7 got unsupported arg(s)" + str(kwargs)
    self.check_chan()
    self.handle = None

  def check_chan(self):
    default = {'gain':1,'offset':0,'make_zero':False,'resolution':1,
               'range':10,'direction':1}
    if not isinstance(self.channels,list):
      self.channels = [self.channels]
    # Let's loop over all the channels to set everything we need
    self.in_chan_list = []
    self.out_chan_list = []
    for d in self.channels:
      if isinstance(d,str):
        d = {'name':d}

      # === AIN channels ===
      if d['name'].startswith("AIN"):
        for k in ['gain','offset','make_zero','resolution','range']:
          if not k in d:
            d[k] = default[k]
        if not 'to_write' in d:
          d['to_write'] = []
        d['to_write'].extend([ # What will be written when opening the chan
            ("_RANGE",d['range']),
            ("_RESOLUTION_INDEX",d['resolution']),
            ])
        if 'thermocouple' in d:
          therm = {'E':20,'J':21,'K':22,'R':23,'T':24,'S':25,'C':30}
          d['to_write'].extend([
              ("_EF_INDEX",therm[d['thermocouple']]),
              ("_EF_CONFIG_A", 1),  # for degrees C
              ("_EF_CONFIG_B", 60052),  # CJC config
            ])
          d['to_read'] = d['name']+"_EF_READ_A"
        elif d["gain"] == 1 and d['offset'] == 0 and not d['make_zero']:
          # No gain/offset
          d['to_read'] = d['name'] # We can read directly of the AIN register
        else: # With gain and offset: let's use Labjack's built in slope
          d['to_write'].extend([
              ("_EF_INDEX",1),
              ("_EF_CONFIG_D",d['gain']),
              ]) # To configure slope in the device
          if not d['make_zero']:
            d['to_write'].append(("_EF_CONFIG_E",d['offset']))
          d['to_read'] = d['name']+"_EF_READ_A" # And read the computed value

        self.in_chan_list.append(d)

      # === DAC/TDAC channels ===
      elif "DAC" in d['name']:
        for k in ['gain','offset']:
          if not k in d:
            d[k] = default[k]
        self.out_chan_list.append(d)

      # === FIO/EIO/CIO/MIO channels ===
      elif "IO" in d['name']:
        if not "direction" in d:
          d["direction"] = default["direction"]
        if d["direction"]: # 1/True => output, 0/False => input
          d['gain'] = 1
          d['offset'] = 0
          self.out_chan_list.append(d)
        else:
          d["to_read"] = d["name"]
          self.in_chan_list.append(d)

      self.in_chan_dict = {}
      for c in self.in_chan_list:
        self.in_chan_dict[c["name"]] = c
      self.out_chan_dict = {}
      for c in self.out_chan_list:
        self.out_chan_dict[c["name"]] = c

  def open(self):
    self.handle = ljm.openS(self.device,self.connection,self.identifier)
    names,values = [],[]
    for c in self.in_chan_list+self.out_chan_list:
      if "to_write" in c:
        for n,v in c['to_write']:
          names.append(c['name']+n)
          values.append(v)
    ljm.eWriteNames(self.handle,len(names),names,values)
    if any([c.get("make_zero",False) for c in self.in_chan_list]):
      print("[Labjack] Please wait during offset evaluation...")
      off = self.eval_offset()
      names,values = [],[]
      for i,c in enumerate(self.in_chan_list):
        if 'make_zero' in c and c['make_zero']:
          names.append(c['name']+'_EF_CONFIG_E')
          values.append(c['offset']+off[i])
      ljm.eWriteNames(self.handle,len(names),names,values)


  def get_data(self):
    """
    Read the signal on all pre-defined input channels.
    """
    try:
      l = [time()]
      l.extend(ljm.eReadNames(self.handle, len(self.in_chan_list),
                              [c['to_read'] for c in self.in_chan_list]))
      return l
    except ljm.LJMError as e:
      print('[Labjack] Error in get_data:', e)
      self.close()
      raise

  def set_cmd(self, *cmd):
    """
    Convert the tension value to a digital value and send it to the output.
    """
    names, values = [],[]
    for val,chan in zip(cmd,self.out_chan_list):
      names.append(chan['name'])
      values.append(chan['gain']*val+chan['offset'])
    ljm.eWriteNames(self.handle,len(names),names,values)

  def __getitem__(self,chan):
    """
    Allows reading of an intput chan by calling lj[chan]
    """
    return time(),ljm.eReadName(self.handle,self.in_chan_dict[chan]['to_read'])

  def __setitem__(self,chan,val):
    """
    Allows setting of an output chan by calling lj[chan] = val
    """
    ljm.eWriteName(self.handle,chan,
        self.out_chan_dict[chan]['gain']*val+self.out_chan_dict[chan]['offset'])

  def write(self,value,address,dtype=ljm.constants.FLOAT32):
    """
    To write data directly into a register
    """
    ljm.eWriteAddress(self.handle,address,dtype,value)

  def close(self):
    """
    Close the device.
    """
    ljm.close(self.handle)
