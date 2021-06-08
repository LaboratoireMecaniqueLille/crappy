# coding: utf-8

from time import time

from .inout import InOut
from .._global import OptionalModule
try:
  from labjack import ljm
except (ModuleNotFoundError, ImportError):
  ljm = OptionalModule("ljm",
      "Please install Labjack LJM and the ljm Python module")


def clamp(val, mini, maxi):
  return max(min(val, maxi), mini)


class Labjack_t7(InOut):
  """Class for LabJack T7 devices.

  It can use any channel as input/output, it can be used with an IOBlock.

  This class is NOT capable of streaming. For higher frequency, see
  :ref:`T7 Streamer` class.

  The keyword argument ``channels`` is used to specify the channels.

  Each channel must be represented as a :obj:`dict` including all the
  parameters. See below for more details on the parameters of the channels.
  """

  def __init__(self,
               device='ANY',
               connection='ANY',
               identifier='ANY',
               channels=None,
               write_at_open=None,
               no_led=False):
    """Sets the args and initializes the parent class.

    Args:
      device (:obj:`str`, optional): The type of the device to open. Possible
        values include:
        ::

          'ANY', 'T7', 'T4', 'DIGIT'

        Only tested with `'T7'` in Crappy.
      connection (:obj:`str`, optional): How is the Labjack connected ?
        Possible values include:
        ::

          'ANY', 'TCP', 'USB', 'ETHERNET', 'WIFI'

      identifier (:obj:`str`, optional): Something to identify the Labjack. It
        can be a serial number, an IP address, or a device name.
      channels (:obj:`list`, optional): Channels to use and their settings. It
        must be a :obj:`list` of :obj:`dict`.
      write_at_open (:obj:`list`, optional): If you need to write specific
        names or registers when opening the channel, you can give them as a
        :obj:`list` of :obj:`tuple`. They will be written in the order of the
        list.
      no_led (:obj:`list`, optional): If :obj:`True`, turns off the LED on the
        Labjack. This led can cause noise on the channels `AIN0` and `AIN1`.

    Note:
      - ``channels`` keys:

        - name (:obj:`str`): The name of the channel according to Labjack's
          naming convention. Ex: `'AIN0'`. This will be used to define the
          direction (in/out) and the available settings.

          It can be:
            - `AINx`: An analog input, if gain and/or offset is given, the
              integrated slope mechanism will be used with the extended
              features registers. It can also be used for thermocouples (see
              below). You can use any EF by using the ``write_at_open`` and
              ``to_read`` keys if necessary.
            - `(T)DACx`: An analog output, you can specify gain and/or offset.
            - `(E/F/C/M IOx)`: Digital in/outputs. You can specify the
              direction.

        - gain (:obj:`float`, default: `1`): A numeric value that will multiply
          the given value for inputs and outputs.
        - offset (:obj:`float`, default: `0`): Will be added to the value.

          For inputs:
          ::

            returned_value = gain * measured_value + offset

          For outputs:
          ::

            set_value = gain * given_value + offset.

          Where `measured_value` and `set_values` are in Volts.

        - make_zero (:obj:`bool`): AIN only, if :obj:`True` the input value
          will be evaluated at startup and the offset will be adjusted to
          return `0` (or the offset if any).
        - direction (:obj:`bool`): DIO only, if :obj:`True` (or `1`), the port
          will be used as an output else as an input.
        - resolution (:obj:`int`, default: `1`): The resolution of the
          acquisition, see Labjack documentation for more details. The bigger
          this value the better the resolution, but the lower the speed. The
          possible range is either `1` to `8` or to `12` according to the
          model.
        - range (:obj:`float`, default: `10`): The range of the acquisition in
          Volts. A range of `x` means that values can be read  between `-x` and
          `x` Volts. The possible values are:
          ::

            0.01, 0.1, 1, 10

        - limits (:obj:`tuple`, default: None): To clamp the output values
          to a given range. The :obj:`tuple` should contain two values: the min
          and the max limit.

        - thermocouple (:obj:`str`): The type of thermocouple (AIN only).
          Possible values are:
          ::

            'E', 'J', 'K', 'R', 'T', 'S', 'C'

          If specified, it will use the EF to read a temperature directly from
          the thermocouples.

        - write_at_open (:obj:`list`): If you need to write specific names or
          registers when opening the channel, you can give them as a
          :obj:`list` of :obj:`tuple`. They will be written in the order of the
          list.

          The tuples can either be `(name (str), value (int/float))` or
          `(register (int), type (int), value (float/int))`.

    Warning:
      DO NOT CONSIDER the ``limits`` KEY AS A SAFETY IMPLEMENTATION. It
      *should* not go beyond/below the given values, but this is not meant to
      replace hardware safety!
    """

    InOut.__init__(self)
    self.device = device
    self.connection = connection
    self.identifier = identifier
    self.channels = [{'name': 'AIN0'}] if channels is None else channels
    self.write_at_open = [] if write_at_open is None else write_at_open
    self.no_led = no_led

    if self.no_led:
      self.write_at_open.append(('POWER_LED', 0))
    self.check_chan()
    self.handle = None

  def check_chan(self):
    default = {'gain': 1, 'offset': 0, 'make_zero': False, 'resolution': 1,
               'range': 10, 'direction': 1, 'dtype': ljm.constants.FLOAT32,
               'limits': None}
    if not isinstance(self.channels, list):
      self.channels = [self.channels]
    # Let's loop over all the channels to set everything we need
    self.in_chan_list = []
    self.out_chan_list = []
    for d in self.channels:
      if isinstance(d, str):
        d = {'name': d}

      # === Modbus registers ===
      if isinstance(d['name'], int):
        for k in ['direction', 'dtype', 'limits']:
          if k not in d:
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
        for k in ['gain', 'offset', 'make_zero', 'resolution', 'range']:
          if k not in d:
            d[k] = default[k]
        if 'write_at_open' not in d:
          d['write_at_open'] = []
        d['write_at_open'].extend([  # What is written when opening the chan
          ljm.nameToAddress(d['name'] + "_RANGE") + (d['range'],),
          ljm.nameToAddress(d['name'] + "_RESOLUTION_INDEX") +
          (d['resolution'],)])
        if 'thermocouple' in d:
          therm = {'E': 20, 'J': 21, 'K': 22, 'R': 23, 'T': 24,
                   'S': 25, 'C': 30}
          d['write_at_open'].extend([
            ljm.nameToAddress(d['name'] + "_EF_INDEX") +
            (therm[d['thermocouple']],),
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_A") + (1,),
            # for degrees C
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_B") + (60052,),
            # CJC config
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_D") + (1,),  # CJC config
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_E") + (0,)  # CJC config
            ])
          d['to_read'], d['dtype'] = ljm.nameToAddress(d['name'] +
                                                       "_EF_READ_A")
        elif d["gain"] == 1 and d['offset'] == 0 and not d['make_zero']:
          # No gain/offset
          # We can read directly of the AIN register
          d['to_read'], d['dtype'] = ljm.nameToAddress(d['name'])
        else:  # With gain and offset: let's use Labjack's built in slope
          d['write_at_open'].extend([
            ljm.nameToAddress(d['name'] + "_EF_INDEX") + (1,),  # for slope
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_D") + (d['gain'],),
            ljm.nameToAddress(d['name'] + "_EF_CONFIG_E") +
            (d['offset'] if not d['make_zero'] else 0,),
          ])  # To configure slope in the device
          d['to_read'], d['dtype'] = ljm.nameToAddress(d['name'] +
                                                       "_EF_READ_A")

        self.in_chan_list.append(d)

      # === DAC/TDAC channels ===
      elif "DAC" in d['name']:
        for k in ['gain', 'offset', 'limits']:
          if k not in d:
            d[k] = default[k]
        d['to_write'], d['dtype'] = ljm.nameToAddress(d['name'])
        self.out_chan_list.append(d)

      # === FIO/EIO/CIO/MIO channels ===
      elif "IO" in d['name']:
        if "direction" not in d:
          d["direction"] = default["direction"]
        if d["direction"]:  # 1/True => output, 0/False => input
          d['gain'] = 1
          d['offset'] = 0
          d['limits'] = None
          d['to_write'], d['dtype'] = ljm.nameToAddress(d['name'])
          self.out_chan_list.append(d)
        else:
          d['to_read'], d['dtype'] = ljm.nameToAddress(d['name'])
          self.in_chan_list.append(d)

      else:
        raise AttributeError("[labjack] Invalid chan name: " + str(d['name']))

      self.in_chan_dict = {}
      for c in self.in_chan_list:
        self.in_chan_dict[c["name"]] = c
      self.out_chan_dict = {}
      for c in self.out_chan_list:
        self.out_chan_dict[c["name"]] = c

  def open(self):
    self.handle = ljm.openS(self.device, self.connection, self.identifier)
    # ==== Writing initial config ====
    reg, types, values = [], [], []
    for t in self.write_at_open:
      if len(t) == 2:
        r, typ = ljm.nameToAddress(t[0])
        value = t[1]
      else:
        r, typ, value = t
      reg.append(r)
      types.append(typ)
      values.append(value)
    for c in self.in_chan_list + self.out_chan_list:
      # Turn (name, val) tuples to (addr, type, val)
      for i, t in enumerate(c.get('write_at_open', [])):
        if len(t) == 2:
          c['write_at_open'][i] = ljm.nameToAddress(t[0]) + (t[1],)
      # Write everything we need
      for r, t, v in c.get('write_at_open', []):
        reg.append(r)
        types.append(t)
        values.append(v)

    if reg:
      ljm.eWriteAddresses(self.handle, len(reg), reg, types, values)
    # ==== Recap of the addresses to read/write ====
    self.read_addresses = [c['to_read'] for c in self.in_chan_list]
    self.read_types = [c['dtype'] for c in self.in_chan_list]
    self.write_addresses = [c['to_write'] for c in self.out_chan_list]
    self.write_types = [c['dtype'] for c in self.out_chan_list]
    self.last_values = [None] * len(self.write_addresses)
    # ==== Measuring zero to add to the offset (if asked to) ====
    if any([c.get("make_zero", False) for c in self.in_chan_list]):
      print("[Labjack] Please wait during offset evaluation...")
      off = self.eval_offset()
      names, values = [], []
      for i, c in enumerate(self.in_chan_list):
        if 'make_zero' in c and c['make_zero']:
          names.append(c['name'] + '_EF_CONFIG_E')
          values.append(c['offset'] + off[i])
      ljm.eWriteNames(self.handle, len(names), names, values)

  def get_data(self):
    """Read the signal on all pre-defined input channels."""

    try:
      return [time()]+ljm.eReadAddresses(self.handle, len(self.read_addresses),
          self.read_addresses, self.read_types)
    except ljm.LJMError as e:
      print('[Labjack] Error in get_data:', e)
      self.close()
      raise

  def set_cmd(self, *cmd):
    """Converts the tension value to a digital value and send it to the output.

    Note:
      Once a value has been written, it will not be written again until it
      changes! This is meant to lower the communication and Labjack activity.

      It relies on the fact that these registers will not change between writes
      (Which is true unless the card runs a lua script writing the same
      registers as the user).
    """

    # values = []
    # for val,chan in zip(cmd, self.out_chan_list):
    #   values.append(chan['gain'] * val + chan['offset'])
    # ljm.eWriteAddresses(self.handle,len(self.write_addresses),
    #    self.write_addresses,self.write_types,values)
    addresses, types, values = [], [], []
    for i, (a, t, v, o, c) in enumerate(zip(self.write_addresses,
                                            self.write_types, cmd,
                                            self.last_values,
                                            self.out_chan_list)):
      if v != o:
        new_v = c['gain'] * v + c['offset']
        if c['limits']:
          new_v = clamp(new_v, c['limits'][0], c['limits'][1])
        self.last_values[i] = v
        addresses.append(a)
        types.append(t)
        values.append(new_v)
    if addresses:
      ljm.eWriteAddresses(self.handle, len(addresses), addresses, types,
                          values)

  def __getitem__(self, chan):
    """Allows reading of an input chan by calling ``lj[chan]``."""

    # Apply offsets and stuff if this is a channel we know
    try:
      return time(), ljm.eReadName(
          self.handle, self.in_chan_dict[chan]['to_read'])
    # Else: let the user access it directly
    except KeyError:
      return time(), ljm.eReadName(self.handle, chan)

  def __setitem__(self, chan, val):
    """Allows setting of an output chan by calling ``lj[chan] = val``."""

    try:
      ljm.eWriteName(self.handle, chan,
       self.out_chan_dict[chan]['gain']*val+self.out_chan_dict[chan]['offset'])
    except KeyError:
      ljm.eWriteName(self.handle, chan, val)

  def write(self, value, address, dtype=None):
    """To write data directly into a register."""

    if dtype is None:
      dtype = ljm.constants.FLOAT32
    ljm.eWriteAddress(self.handle, address, dtype, value)

  def read(self, address, dtype=None):
    """To read data directly from a register."""

    if dtype is None:
      dtype = ljm.constants.FLOAT32
    return ljm.eReadAddress(self.handle, address, dtype)

  def close(self):
    """Closes the device."""

    ljm.close(self.handle)
