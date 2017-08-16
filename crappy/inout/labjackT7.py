# coding: utf-8

from labjack import ljm
from time import time

from .inout import InOut


class Labjack_t7(InOut):
  """
  Class for LabJack T7 devices.

  It can use any channel as input/output, it can be used with an IOBlock.
  This class is NOT capable of streaming. For higher frequency, see
  T7_streamer class.
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
        EF by using the 'write_at_open' and 'to_read' keys if necessary.
      - (T)DACx: An analog output, you can specifiy gain and/or offset.
      - (E/F/C/M IOx): Digital in/outputs. You can specify the direction.

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
        'range':10,'direction':1,'dtype':ljm.constants.FLOAT32}
    if not isinstance(self.channels,list):
      self.channels = [self.channels]
    # Let's loop over all the channels to set everything we need
    self.in_chan_list = []
    self.out_chan_list = []
    for d in self.channels:
      if isinstance(d,str):
        d = {'name':d}

      # === Modbus registers ===
      if isinstance(d['name'],int):
        for k in ['direction','dtype']:
          if not k in d:
            d[k] = default[k]
        if d['direction']:
          d['to_write'] = d['name']
          d['gain'] = 1
          d['offset'] = 0
          self.out_chan_list.append(d)
        else:
          d['to_read'] = d['name']
          self.in_chan_list.append(d)

      # === AIN channels ===
      elif d['name'].startswith("AIN"):
        for k in ['gain','offset','make_zero','resolution','range']:
          if not k in d:
            d[k] = default[k]
        if not 'write_at_open' in d:
          d['write_at_open'] = []
        d['write_at_open'].extend([ # What will be written when opening the chan
            ljm.nameToAddress(d['name']+"_RANGE")+(d['range'],),
            ljm.nameToAddress(d['name']+"_RESOLUTION_INDEX")+(d['resolution'],)
            ])
        if 'thermocouple' in d:
          therm = {'E':20,'J':21,'K':22,'R':23,'T':24,'S':25,'C':30}
          d['write_at_open'].extend([
              ljm.nameToAddress(d['name']+"_EF_INDEX")\
                +(therm[d['thermocouple']],),
              ljm.nameToAddress(d['name']+"_EF_CONFIG_A")+(1,), # for degrees C
              ljm.nameToAddress(d['name']+"_EF_CONFIG_B")+(60052,) # CJC config
            ])
          d['to_read'],d['dtype'] = ljm.nameToAddress(d['name']+"_EF_READ_A")
        elif d["gain"] == 1 and d['offset'] == 0 and not d['make_zero']:
          # No gain/offset
          # We can read directly of the AIN register
          d['to_read'],d['dtype'] = ljm.nameToAddress(d['name'])
        else: # With gain and offset: let's use Labjack's built in slope
          d['write_at_open'].extend([
              ljm.nameToAddress(d['name']+"_EF_INDEX")+(1,), # for slope
              ljm.nameToAddress(d['name']+"_EF_CONFIG_D")+(d['gain'],),
              ljm.nameToAddress(d['name']+"_EF_CONFIG_E")\
                  +(d['offset'] if not d['make_zero'] else 0,),
              ]) # To configure slope in the device
          d['to_read'],d['dtype'] = ljm.nameToAddress(d['name']+"_EF_READ_A")

        self.in_chan_list.append(d)

      # === DAC/TDAC channels ===
      elif "DAC" in d['name']:
        for k in ['gain','offset']:
          if not k in d:
            d[k] = default[k]
        d['to_write'],d['dtype'] = ljm.nameToAddress(d['name'])
        self.out_chan_list.append(d)

      # === FIO/EIO/CIO/MIO channels ===
      elif "IO" in d['name']:
        if not "direction" in d:
          d["direction"] = default["direction"]
        if d["direction"]: # 1/True => output, 0/False => input
          d['gain'] = 1
          d['offset'] = 0
          d['to_write'],d['dtype'] = ljm.nameToAddress(d['name'])
          self.out_chan_list.append(d)
        else:
          d['to_read'],d['dtype'] = ljm.nameToAddress(d['name'])
          self.in_chan_list.append(d)

      else:
        raise AttributeError("[labjack] Invalid chan name: "+str(d['name']))

      self.in_chan_dict = {}
      for c in self.in_chan_list:
        self.in_chan_dict[c["name"]] = c
      self.out_chan_dict = {}
      for c in self.out_chan_list:
        self.out_chan_dict[c["name"]] = c

  def open(self):
    self.handle = ljm.openS(self.device,self.connection,self.identifier)
    # ==== Writing initial config ====
    reg,types,values = [],[],[]
    for c in self.in_chan_list+self.out_chan_list:
      for r,t,v in c.get('write_at_open',[]):
        reg.append(r)
        types.append(t)
        values.append(v)
    if reg:
      ljm.eWriteAddresses(self.handle,len(reg),reg,types,values)
    # ==== Recap of the addresses to read/write ====
    self.read_addresses = [c['to_read'] for c in self.in_chan_list]
    self.read_types = [c['dtype'] for c in self.in_chan_list]
    self.write_addresses = [c['to_write'] for c in self.out_chan_list]
    self.write_types = [c['dtype'] for c in self.out_chan_list]
    self.last_values = [None]*len(self.write_addresses)
    # ==== Measuring zero to add to the offset (if asked to) ====
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
      return [time()]+ljm.eReadAddresses(self.handle, len(self.read_addresses),
          self.read_addresses,self.read_types)
    except ljm.LJMError as e:
      print('[Labjack] Error in get_data:', e)
      self.close()
      raise

  def set_cmd(self, *cmd):
    """
    Convert the tension value to a digital value and send it to the output.

    Note that once a value has been written, it will not be written again until
    it changes! This is meant to lower the communication and Labjack activity.
    It relies on the fact that these registers will not change between writes.
    (Which is true unless the card runs a lua script writing the same registers
    as the user).
    """
    #values = []
    #for val,chan in zip(cmd,self.out_chan_list):
    #  values.append(chan['gain']*val+chan['offset'])
    #ljm.eWriteAddresses(self.handle,len(self.write_addresses),
    #    self.write_addresses,self.write_types,values)
    addresses,types,values = [],[],[]
    for i,(a,t,v,o,c) in enumerate(zip(
        self.write_addresses,self.write_types,
        cmd,self.last_values,self.out_chan_list)):
      if v != o:
        new_v = c['gain']*v+c['offset']
        self.last_values[i] = v
        addresses.append(a)
        types.append(t)
        values.append(new_v)
    if addresses:
      ljm.eWriteAddresses(self.handle,len(addresses),addresses,types,values)

  def __getitem__(self,chan):
    """
    Allows reading of an input chan by calling lj[chan]
    """
    # Apply offsets and stuff if this is a channel we know
    try:
      return time(),ljm.eReadName(
          self.handle,self.in_chan_dict[chan]['to_read'])
    # Else: let the user access it directly
    except KeyError:
      return time(),ljm.eReadName(self.handle,chan)

  def __setitem__(self,chan,val):
    """
    Allows setting of an output chan by calling lj[chan] = val
    """
    try:
      ljm.eWriteName(self.handle,chan,
        self.out_chan_dict[chan]['gain']*val+self.out_chan_dict[chan]['offset'])
    except KeyError:
      ljm.eWriteName(self.handle,chan,val)

  def write(self,value,address,dtype=ljm.constants.FLOAT32):
    """
    To write data directly into a register
    """
    ljm.eWriteAddress(self.handle,address,dtype,value)

  def read(self,address,dtype=ljm.constants.FLOAT32):
    """
    To read data directly from a register
    """
    return ljm.eReadAddress(self.handle,address,dtype)

  def close(self):
    """
    Close the device.
    """
    ljm.close(self.handle)
