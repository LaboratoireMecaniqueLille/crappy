# coding: utf-8

import subprocess
from threading import Thread, RLock
import time
from typing import Union, Dict
from .actuator import Actuator
from .._global import OptionalModule

try:
  import yaml
except (ModuleNotFoundError, ImportError):
  yaml = OptionalModule("pyyaml")

try:
  import usb.core
  import usb.util

  Tic_usb_request = {'Cmd': usb.util.CTRL_OUT |
                            usb.util.CTRL_TYPE_VENDOR |
                            usb.util.CTRL_RECIPIENT_DEVICE,
                     'Var': usb.util.CTRL_IN |
                            usb.util.CTRL_TYPE_VENDOR |
                            usb.util.CTRL_RECIPIENT_DEVICE}
except (ModuleNotFoundError, ImportError):
  usb = OptionalModule("pyusb")
  Tic_usb_request = {'Cmd': 0x40,
                     'Var': 0xC0}

Tic_vendor_id = 0x1FFB

Tic_product_id = {'T825': 0x00B3,
                  'T834': 0x00B5,
                  'T500': 0x00BD,
                  'N825': 0x00C3,
                  'T249': 0x00C9,
                  '36v4': 0x00CB}

Tic_max_allowed_current = {'T825': 3968,
                           'T834': 3456,
                           'T500': 3093,
                           'T249': 4480,
                           '36v4': 3939}

Tic_36v4_max_current = 9095

Tic_current_steps = {'T834': 32,
                     'T825': 32,
                     'T249': 40,
                     '36v4': 71.615}

Tic_step_modes = {'T825': [2 ** i for i in range(6)],
                  'T834': [2 ** i for i in range(6)],
                  'T500': [2 ** i for i in range(4)],
                  'T249': [2 ** i for i in range(6)] + ['2_100p'],
                  '36v4': [2 ** i for i in range(9)]}

Tic_step_mode = {1: 0,
                 2: 1,
                 4: 2,
                 8: 3,
                 16: 4,
                 32: 5,
                 '2_100p': 6,
                 64: 7,
                 128: 8,
                 256: 9}

Tic_cmd = {'Set_target_position': 0xE0,
           'Set_target_velocity': 0xE3,
           'Halt_and_set_position': 0xEC,
           'Halt_and_hold': 0x89,
           'Go_home': 0x97,
           'Reset_command_timeout': 0x8C,
           'Deenergize': 0x86,
           'Energize': 0x85,
           'Exit_safe_start': 0x83,
           'Enter_safe_start': 0x8F,
           'Reset': 0xB0,
           'Clear_driver_error': 0x8A,
           'Set_max_speed': 0xE6,
           'Set_starting_speed': 0xE5,
           'Set_max_accel': 0xEA,
           'Set_max_decel': 0xE9,
           'Set_step_mode': 0x94,
           'Set_current_limit': 0x91,
           'Set_decay_mode': 0x92,
           'Set_AGC_option': 0x98,
           'Get_variable': 0xA1,
           'Get_variable_and_clear_errors_occurred': 0xA2,
           'Get_setting': 0xA8,
           'Set_setting': 0x13,
           'Reinitialize': 0x10,
           'Start_bootloader': 0xFF,
           'Get_debug_data': 0x20}

# offsets/indexes
Tic_var = {'Operation_state': 0x00,
           'Misc_flags1': 0x01,
           'Error_status': 0x02,
           'Errors_occurred': 0x04,
           'Planning_mode': 0x09,
           'Target_position': 0x0A,
           'Target_velocity': 0x0E,
           'Starting_speed': 0x12,
           'Max_speed': 0x16,
           'Max_decel': 0x1A,
           'Max_accel': 0x1E,
           'Current_position': 0x22,
           'Current_velocity': 0x26,
           'Acting_target_position': 0x2A,
           'Time_since_last_step': 0x2E,
           'Device_reset': 0x32,
           'Vin_voltage': 0x33,
           'Up_time': 0x35,
           'Encoder_position': 0x39,
           'RC_pulse_width': 0x3D,
           'Analog_reading_SCL': 0x3F,
           'Analog_reading_SDA': 0x41,
           'Analog_reading_TX': 0x43,
           'Analog_reading_RX': 0x45,
           'Digital_readings': 0x47,
           'Pin_states': 0x48,
           'Step_mode': 0x49,
           'Current_limit': 0x4A,
           'Decay_mode': 0x4B,
           'Input_state': 0x4C,
           'Input_after_averaging': 0x4D,
           'Input_after_hysteresis': 0x4F,
           'Input_after_scaling': 0x51}

# indexes
Tic_settings = {'Setting_not_initialized': 0x00,
                'Control_mode': 0x01,
                'Never_sleep': 0x02,
                'Disable_safe_start': 0x03,
                'Ignore_err_line_high': 0x04,
                'Serial_baud_rate_generator': 0x05,
                'Serial_device_number': 0x07,
                'Auto_clear_driver_error': 0x08,
                'Command_timeout_low': 0x09,
                'Command_timeout_high': 0x0A,
                'Serial_CRC_enabled': 0x0B,
                'Low_vin_timeout': 0x0C,
                'Low_vin_shutoff_voltage': 0x0E,
                'Low_vin_startup_voltage': 0x10,
                'High_vin_shutoff_voltage': 0x12,
                'Vin_calibration': 0x14,
                'RC_max_pulse_period': 0x16,
                'RC_bad_signal_timeout': 0x18,
                'RC_consecutive_good_pulses': 0x1A,
                'Invert_motor_direction': 0x1B,
                'Input_error_min': 0x1C,
                'Input_error_max': 0x1E,
                'Input_scaling_degree': 0x20,
                'Input_invert': 0x21,
                'Input_min': 0x22,
                'Input_neutral_min': 0x24,
                'Input_neutral_max': 0x26,
                'Input_max': 0x28,
                'Output_min': 0x2A,
                'Input_averaging_enabled': 0x2E,
                'Input_hysteresis': 0x2F,
                'Current_limit_during_error': 0x31,
                'Output_max': 0x32,
                'Switch_polarity_map': 0x36,
                'Encoder_postscaler': 0x37,
                'SCL_config': 0x3B,
                'SDA_config': 0x3C,
                'TX_config': 0x3D,
                'RX_config': 0x3E,
                'RC_config': 0x3F,
                'Current_limit': 0x40,
                'Step_mode': 0x41,
                'Decay_mode': 0x42,
                'Starting_speed': 0x43,
                'Max_speed': 0x47,
                'Max_decel': 0x4B,
                'Max_accel': 0x4F,
                'Soft_error_response': 0x53,
                'Soft_error_position': 0x54,
                'Encoder_prescaler': 0x58,
                'Encoder_unlimited': 0x5C,
                'Kill_switch_map': 0x5D,
                'Serial_response_delay': 0x5E,
                'Limit_switch_forward_map': 0x5F,
                'Limit_switch_reverse_map': 0x60,
                'Homing_speed_towards': 0x61,
                'Homing_speed_away': 0x65,
                'Serial_device_number_high': 0x69,
                'Serial_alt_device_number': 0x6A,
                'Size': 0x5F,
                'Unrestricted_current_limit': 0x6C}

Tic_current_tables = {'T500': [0, 1, 174, 343, 495, 634, 762, 880, 990, 1092,
                               1189, 1281, 1368, 1452, 1532, 1611, 1687, 1762,
                               1835, 1909, 1982, 2056, 2131, 2207, 2285, 2366,
                               2451, 2540, 2634, 2734, 2843, 2962, 3093],
                      'T834': [index * Tic_current_steps['T834'] for index in
                               (list(range(33)) + list(range(34, 65, 2)) +
                                list(range(68, 109, 4)))],
                      'T825': [index * Tic_current_steps['T825'] for index in
                               (list(range(33)) + list(range(34, 65, 2)) +
                                list(range(68, 125, 4)))],
                      'T249': [index * Tic_current_steps['T249'] for index in
                               (list(range(33)) + list(range(34, 65, 2)) +
                                list(range(68, 113, 4)))],
                      '36v4': [index * Tic_current_steps['36v4'] for index in
                               range(128)]}

Tic_max_accel = 2147483647  # steps/s/100s
Tic_min_accel = 100  # steps/s/100s
Tic_max_speed = 500000000  # steps/10000s, i.e. a 50 kHz frequency
Tic_min_speed = 7  # steps/10000s, i.e. 1 step every 23 minutes

Tic_backends = ['ticcmd', 'USB']

Tic_pins_bit = {'SCL': 0,
                'SDA': 1,
                'TX': 2,
                'RX': 3,
                'RC': 4}
Tic_pin_modes = {'Default': 0,
                 'Kill switch': 7,
                 'Limit switch forward': 8,
                 'Limit switch reverse': 9}
Tic_pin_polarity = {'Active low': 0,
                    'Active high': 1}


class Find_serial_number:
  """A class used for finding USB devices matching a given serial number, using
     the :meth:`usb.core.find` method."""

  def __init__(self, serial_number: str) -> None:
    self.serial_number = serial_number

  def __call__(self, device) -> bool:
    return device.serial_number == self.serial_number


class Pololu_tic(Actuator):
  """Class for controlling Pololu's Tic stepper motor divers.

  The Pololu_tic Actuator block is meant for controlling a Pololu Tic stepper
  motor driver. It can be driven in both speed and position. Several Tic models
  are supported. The length unit is the millimeter (`mm`), and time unit is the
  second (`s`).

  Important:
    **Only for Linux users:** In order to drive the Tic, the appropriate udev
    rule should be set. This is done automatically when installing `ticcmd`,
    or can be done using the `udev_rule_setter` utility in ``crappy``'s `util`
    folder. It is also possible to add it manually by running:
    ::

      $ echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"1ffb\\", \
MODE=\\"0666\\\"" | sudo tee pololu.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.
  """

  def __init__(self,
               steps_per_mm: float,
               current_limit: float,
               step_mode: Union[int, str] = 8,
               max_accel: float = 20,
               t_shutoff: float = 0,
               config_file: str = None,
               serial_number: str = None,
               model: str = None,
               reset_command_timeout: bool = True,
               backend: str = 'USB',
               unrestricted_current_limit: bool = False,
               pin_function: Dict[str, str] = None,
               pin_polarity: Dict[str, str] = None) -> None:
    """Checks args validity, finds the right device, reads the current limit
    tables.

    Args:
      steps_per_mm (:obj:`float`): The number of full steps needed for the
        motor to travel `1mm`. This varies according to the motor model, and
        can be deduced from the datasheet or directly measured. This value is
        usually between `50` and `500`.
      current_limit (:obj:`float`): The maximum current the motor is able to
        withstand, in mA. It is usually around `1A` for small stepper motors,
        and can go up to a few Amps. The maximum allowed ``current_limit``
        value depends on the Tic model. The Tic 36v4 default maximum current
        limit can be increased using the ``unrestricted_current_limit``
        parameter.
      step_mode (:obj:`int`, optional): Sets the number of microsteps used for
        driving the motor. This number is always a power of `2`. The minimum
        number of microsteps is `1` (full steps), and the maximum depends on
        the Tic model. All models however support modes `1` to `8`. The block
        manages speed and length conversions so that changing the step mode
        doesn't affect the motor behaviour.
      max_accel (:obj:`float`, optional): The maximum allowed acceleration for
        the motor, in `mm/s²`. When asked to reach a given speed or position,
        the motor accelerates at this rate. It also corresponds to the maximum
        allowed deceleration. Usually doesn't need to be changed.
      t_shutoff (:obj:`float`, optional): The :class:`Pololu_tic` block
        features an auto-shutoff thread that deenergizes the motor after a
        period of `t_shutoff` seconds of inactivity. The timer counts in steps
        of `0.1s`, which is thus the maximum precision for this setting. When
        set to `0`, this feature is disabled and the motor remains energized
        until the :meth:`close` method is called.
      config_file (:obj:`str`, optional): The path of the config file to be
        loaded to the Tic. It only works if ``backend`` is 'ticcmd'. The config
        file contains some specific settings that can only be accessed this way
        using the 'ticcmd' backend. Not necessary for most applications.
      serial_number (:obj:`str`, optional): The serial number of the Tic to be
        controlled. It must be given as a :obj:`str`, and it is an 8-digits
        number. Allows to control the right device if several Tic of the same
        model are connected. Otherwise an error is raised.
      model (:obj:`str`, optional): The model of the Tic to be controlled.
        Available models are:
        ::

          'T825', 'T824', 'T500', 'N825', 'T249', '36v4'

        Allows to control the right device if several Tic of different models
        are connected. Otherwise an error is raised.
      reset_command_timeout (:obj:`bool`, optional): Enables or disables the
        `reset_command_timeout` thread. It can only be disabled if ``backend``
        is 'USB'. This thread pings the Tic every `0.5s`, so that it doesn't
        block due to a Command Timeout error. This feature is a safety to
        prevent the motor from running indefinitely if the USB connection is
        down, so it is better not to disable it. When disabled the Tic never
        raises Command Timeout errors, and a bit of memory if freed because of
        the thread not running.
      backend (:obj:`str`, optional): The backend for communicating with the
        Tic. Available backends are:
        ::

          'USB', 'ticcmd'

        They both communicate over USB, but 'ticcmd' requires Pololu's firmware
        to be installed. Some features are specific to each backend.
      unrestricted_current_limit (:obj:`bool`, optional): Enables or disables
        the unrestricted current limit feature. Only works if ``backend`` is
        'USB', and for the 36v4 Tic model. When disabled, the maximum current
        allowed is `3939mA`. If enabled, it goes up to `9095mA`. The Tic should
        however be cooled in order to withstand currents higher than `3939mA`.
      pin_function (:obj:`dict`, optional): Allows setting the Tic GPIO
        functions. It is a :obj:`dict` whose keys are the pin names, and values
        are the functions. Only works if ``backend`` is `'USB'`. Only the pins
        indicated in ``pin_function`` are set, the others are left in their
        previous state. The available pins are:
        ::

          'SCL', 'SDA', 'TX', 'RX', 'RC'

        and can be set to:
        ::

          'Default', 'Kill switch', 'Limit switch forward', \
'Limit switch reverse'

        The GPIO functions remain set as long as they are not changed by the
        user, so for a given setup it is only necessary to set them once.
      pin_polarity (:obj:`dict`, optional): Allows setting the polarity of the
        GPIOs used as switches. It is a :obj:`dict`, whose keys are the pin
        names, and values are the pin polarities. Only works if ``backend`` is
        `'USB'`. Only the pins indicated in ``pin_function`` are set, the
        others are left in their previous state. The available pins are:
        ::

          'SCL', 'SDA', 'TX', 'RX', 'RC'

        and can be set to:
        ::

          'Active high', 'Active low'

        The GPIO polarities remain set as long as they are not changed by the
        user, so for a given setup it is only necessary to set them once.

    Warning:
      - ``current_limit``:
        If the ``current_limit`` setting is higher than the motor max current,
        there's a risk of overheating and damaging the motor !

    Note:
      - ``steps_per_mm``:
        If you have to measure this value, it can be done easily following this
        procedure. Set ``steps_per_mm`` to `spm` (`100` should be fine), and
        ``step_mode`` to `sm` (`8` should be fine). Run a crappy program
        for moving the motor from position `0` to position `p` (a few tenth of
        millimeters should be fine). The motor will reach an actual position
        `ap` that can be measured. The actual ``steps_per_mm`` value `aspm` for
        this motor can be calculated as follows:
        ::

          aspm = spm * p / ap

      - ``step_mode``:
        Increasing the number of microsteps allows to reduce the noise, the
        vibrations, and improve the precision. However the more microsteps, the
        lower the maximum achievable speed for the motor. Chances that the
        motor misses microsteps are also higher when the number of microsteps
        is high.
      - ``t_shutoff``:
        This functionality was originally added for long assays in temperature
        controlled environments, so that the motor doesn't unnecessarily heat
        the setup when inactive. In other assays, it may still be useful for
        reducing the noise, the electromagnetic interference, or the energy
        consumption.
      - ``serial_number``:
        Serial numbers can be accessed using the `lsusb` command in Linux
        shell, or running ``ticcmd --list`` if `ticcmd` is installed. This
        number is also printed during :meth:`__init__` if only one device is
        connected and ``serial_number`` is :obj:`None`.
      - ``model``:
        The model is written on the Tic board, and can be accessed by running
        ``ticcmd --list`` in a shell if `ticcmd` is installed. It is also
        printed during :meth:`__init__` if only one device is connected and
        ``model`` is :obj:`None`.
      - **Pins settings**:
        The pin functions and polarity can also be set independently from
        ``crappy`` before starting the assay, in the `ticgui`.
    """

    Actuator.__init__(self)

    if backend not in Tic_backends:
      raise ValueError("backend should be in {}".format(Tic_backends))
    else:
      self._backend = backend

    if model is not None and model not in Tic_product_id:
      raise ValueError("model should be in {} if given".format(list(
        Tic_product_id.keys())))

    if serial_number is not None and type(serial_number) is not str:
      raise ValueError("serial_number should be given as a string")

    # Finding the right device among all the connected ones
    if backend == 'USB':
      # Finding all devices matching the given inputs
      if model is None:
        if serial_number is None:
          devices = usb.core.find(find_all=True,
                                  idVendor=Tic_vendor_id)
        else:
          devices = usb.core.find(find_all=True,
                                  idVendor=Tic_vendor_id,
                                  custom_match=Find_serial_number(
                                    serial_number))
      else:
        if serial_number is None:
          devices = usb.core.find(find_all=True,
                                  idVendor=Tic_vendor_id,
                                  idProduct=Tic_product_id[model])
        else:
          devices = usb.core.find(find_all=True,
                                  idVendor=Tic_vendor_id,
                                  idProduct=Tic_product_id[model],
                                  custom_match=Find_serial_number(
                                    serial_number))
      # Making sure there's only one matching device
      devices = list(devices)
      if len(devices) == 0:
        raise IOError("No matching device connected")
      elif len(devices) > 1:
        raise IOError("Several matching devices found, try specifying a "
                      "device or a serial_number")
      else:
        self._dev = devices[0]
      # Setting self.serial_number and self.device
      if serial_number is None:
        self._serial_number = usb.util.get_string(self._dev,
                                                  self._dev.iSerialNumber)
      else:
        self._serial_number = serial_number
      if model is None:
        try:
          self._model = next(key for key, value in Tic_product_id.items() if
                             value == self._dev.idProduct)
        except StopIteration:
          raise ValueError("The Tic model automatically found is not "
                           "implemented in crappy")
      else:
        self._model = model

    elif backend == 'ticcmd':
      # Finding all devices matching the given inputs
      devices = subprocess.check_output(['ticcmd'] + ['--list']).\
        decode("utf-8").split("\n")
      devices.pop()  # Removing the '' element at the end of devices
      devices = [string.split(',') for string in devices]
      if model is not None:
        if serial_number is not None:
          devices = [dev for dev in devices if dev[0] == serial_number and
                     model in dev[1]]
        else:
          devices = [dev for dev in devices if model in dev[1]]
      elif serial_number is not None:
        devices = [dev for dev in devices if dev[0] == serial_number]
      # Making sure there's only one matching device
      if len(devices) == 0:
        raise IOError("No matching device found")
      elif len(devices) > 1:
        raise IOError("Several matching devices found, try specifying a "
                      "device or a serial_number")
      # Setting self.serial_number and self.device
      if serial_number is None:
        self._serial_number = devices[0][0]
      else:
        self._serial_number = serial_number
      if model is None:
        try:
          self._model = next(key for key in Tic_product_id if key in
                             devices[0][1])
        except StopIteration:
          raise ValueError("The Tic model automatically found is not "
                           "implemented in crappy")
      else:
        self._model = model

    # Printing model and serial_number if they were not specified by the user
    if serial_number is None:
      print("Tic serial number :", self._serial_number)
    if model is None:
      print("Tic model :", self._model)

    # Making sure the current limit is valid, especially for the 36v4 model
    if not 0 < current_limit < Tic_max_allowed_current[self._model]:
      if self._model == '36v4':
        if not 0 < current_limit < Tic_36v4_max_current:
          raise ValueError("current_limit should be between 0 and {} mA for "
                           "this Tic model".format(Tic_36v4_max_current))
        elif not unrestricted_current_limit:
          raise ValueError("current_limit exceeds the safety limit, which may "
                           "cause overheating. Set unrestricted_current_limit "
                           "to True if you want to keep this current_limit "
                           "(only works if backend='USB')")
        elif backend != 'USB':
          raise ValueError("Setting unrestricted_current_limit to True only "
                           "works if backend='USB'")
      else:
        raise ValueError("current limit should be between 0 and {} mA "
                         "for this Tic model".
                         format(Tic_max_allowed_current[self._model]))
    self._current_limit = current_limit
    self._unrestricted_current_limit = unrestricted_current_limit

    # Converting the current limit value to a current index, used by the
    # USB backend only
    if backend == 'USB':
      if self._model == 'T500':
        self._current_index = min(enumerate(Tic_current_tables[self._model]),
                                  key=lambda x: abs(x[1] - current_limit))[0]
      else:
        self._current_index = round(min(Tic_current_tables[self._model],
                                    key=lambda x: abs(x - current_limit)) /
                                    Tic_current_steps[self._model])

    if step_mode not in Tic_step_modes[self._model]:
      raise ValueError("step_mode should be in {}".format(
        Tic_step_modes[self._model]))
    else:
      self._step_mode = step_mode

    if steps_per_mm < 0:
      raise ValueError("steps_per_mm should be positive")
    else:
      self._steps_per_mm = steps_per_mm

    # Keeping the max_accel and max_decel values within the Tic ratings
    if max_accel < self._to_mm(Tic_min_accel / 100):
      print(
        "Requested acceleration below min allowed acceleration, "
        "setting to min allowed acceleration")
      max_accel = self._to_mm(Tic_min_accel / 100)
    elif max_accel > self._to_mm(Tic_max_accel / 100):
      print(
        "Requested acceleration exceeding max allowed acceleration, "
        "setting to max allowed acceleration")
      max_accel = self._to_mm(Tic_max_accel / 100)
    self._max_accel = max_accel

    if t_shutoff < 0:
      raise ValueError("t_shutoff should be zero or positive")
    else:
      self._t_shutoff = t_shutoff

    if config_file is not None and backend != 'ticcmd':
      print("Warning : config files can only be loaded if backend='ticcmd', "
            "ignoring the given config_file")
      self._config_file = None
    else:
      self._config_file = config_file

    if backend != 'USB' and not reset_command_timeout:
      print("Warning : reset_command_timeout can only be disabled if "
            "backend='USB', reset_command_timeout set to True")
      self._rct_on = True
    else:
      self._rct_on = reset_command_timeout

    if backend != 'USB' and pin_function is not None:
      raise ValueError("It is not possible to set the pin functions if "
                       "the backend is not 'USB'")
    if pin_function is not None:
      if not all(key in Tic_pins_bit for key in pin_function):
        raise ValueError("Unexpected pin name, pin names should be in "
                         "{}".format(list(Tic_pins_bit.keys())))
      if not all(value in Tic_pin_modes for value in pin_function.values()):
        raise ValueError("Unexpected pin function, pin functions should be in "
                         "{}".format(list(Tic_pin_modes.keys())))
    self._pin_function = pin_function

    if backend != 'USB' and pin_polarity is not None:
      raise ValueError("It is not possible to set the pin polarities if "
                       "the backend is not 'USB'")
    if pin_polarity is not None:
      if not all(key in Tic_pins_bit for key in pin_polarity):
        raise ValueError("Unexpected pin name, pin names should be in "
                         "{}".format(list(Tic_pins_bit.keys())))
      if not all(value in Tic_pin_polarity for value in pin_polarity.values()):
        raise ValueError("Unexpected pin function, pin functions should be in "
                         "{}".format(list(Tic_pin_polarity.keys())))
    self._pin_polarity = pin_polarity

    # The lock is meant for preventing interferences between the threads
    self._lock = RLock()

    # Definition of the flags
    self._timer_shutoff = False
    self._RCT = False
    self._reset_timer_shutoff = False
    self._close = False

    # Definition of the auxiliary threads
    self._thrd_rct = Thread(target=self._thread_rct)
    self._thrd_shutoff = Thread(target=self._thread_shutoff)

  def open(self):
    """Sets the communication, the motor parameters and starts the threads."""

    if self._backend == 'USB':
      try:
        self._dev.set_configuration()
      except usb.core.USBError:
        print("You may have to install the udev-rules for this USB device, "
              "this can be done using the udev_rule_setter utility in the util "
              "folder")
        raise

    # Setting the Tic according to the user parameters
    pin_changed = False
    if self._pin_polarity is not None:
      self._set_pin_polarity(self._pin_polarity)
      pin_changed = True
    if self._pin_function is not None:
      self._set_pin_function(self._pin_function)
      pin_changed = True
    if pin_changed:
      self._reset()

    self._enter_safe_start()
    self._deenergize()
    self._set_step_mode()
    self._set_current_limit()
    self._set_max_accel()
    self._set_max_decel()

    # Loading the config file
    if self._config_file is not None:
      self._ticcmd('--settings', str(self._config_file))

    # Starting the auxiliary threads
    # The RCT thread is not needed in case reset_command_timeout is False
    # The shutoff thread is not needed in case t_shutoff is zero
    if self._rct_on:
      self._thrd_rct.start()
    else:
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=0x00,
                        index=Tic_settings['Command_timeout_low'])
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=0x00,
                        index=Tic_settings['Command_timeout_high'])
    if self._t_shutoff > 0:
      self._thrd_shutoff.start()

  def get_speed(self) -> float:
    """Reads the current motor speed.

        Returns:
          :obj:`float`: The speed in mm/s
        """
    if self._backend == 'ticcmd':
      return self._to_mm(yaml.load(self._ticcmd('-s'), Loader=yaml.FullLoader)
                         ['Current velocity'] / 10000)
    elif self._backend == 'USB':
      return self._to_mm(int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_variable'],
                          index=Tic_var['Current_velocity'],
                          data_or_length=4),
        byteorder='little',
        signed=True) / 10000)

  def get_pos(self) -> float:
    """Reads the current motor position.

    Returns:
      :obj:`float`: The position in mm
    """

    if self._backend == 'ticcmd':
      return self._to_mm(yaml.load(self._ticcmd('-s'), Loader=yaml.FullLoader)
                         ['Current position'])
    elif self._backend == 'USB':
      return self._to_mm(int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_variable'],
                          index=Tic_var['Current_position'],
                          data_or_length=4),
        byteorder='little',
        signed=True))

  def set_position(self, position: float, speed: float = None) -> None:
    """Sends a position command to the motor.

    Args:
      position (:obj:`float`): The position to reach in `mm`
      speed (:obj:`float`, optional): The speed at which the motor should move
        to the given position, in `mm/s`

    Note:
      - ``speed``:
        The only way to reach a position at a given speed is to change the
        maximum speed. The Tic will try to accelerate to the maximum speed but
        may remain slower if it doesn't have time to do so before reaching the
        given position.
    """

    if speed is not None:
      self._set_max_speed(speed)

    # Energizing the motor
    self._energize()
    self._exit_safe_start()

    # Raising the flags
    self._timer_shutoff = True
    self._reset_timer_shutoff = True
    self._RCT = True

    # Sending the position command
    self._set_position(position)

  def set_speed(self, speed: float) -> None:
    """Sends a speed command to the motor.

    Args:
      speed (:obj:`float`): The speed the motor should reach
    """

    # Changing the maximum speed if needed
    max_speed = self._get_max_speed()
    if abs(speed) > max_speed:
      self._set_max_speed(speed)

    # The command speed may first need to be reduced or increased in order to
    # comply with the Tic ratings
    final_speed = min(abs(self._to_steps(speed * 10000)), Tic_max_speed)
    if final_speed:  # If speed is 0, then it should remain 0
      if final_speed < Tic_min_speed:
        print(
          "Requested speed below min possible speed, setting to min "
          "possible speed")
        final_speed = Tic_min_speed
      final_speed *= speed / abs(speed)

    # Energizing the motor
    self._energize()
    self._exit_safe_start()

    # Raising the flags
    self._timer_shutoff = True
    self._reset_timer_shutoff = True
    self._RCT = True

    # Sending the speed command
    self._set_velocity(final_speed)

  def stop(self) -> None:
    """Sets the speed to `0`."""

    self._set_velocity(0)

  def close(self) -> None:
    """Stops the motor, joins the threads and deenergizes the motor."""

    self.stop()
    self._close = True
    if self._rct_on:
      self._thrd_rct.join()
    if self._t_shutoff > 0:
      self._thrd_shutoff.join()
    self._enter_safe_start()
    self._deenergize()
    if self._backend == 'USB':
      usb.util.dispose_resources(self._dev)

  def _to_steps(self, mm: float) -> float:
    """Wrapper for converting `mm` to `steps`."""

    return mm * self._steps_per_mm * self._step_mode

  def _to_mm(self, steps: float) -> float:
    """Wrapper for converting `steps` to `mm`."""

    return steps / self._steps_per_mm / self._step_mode

  def _reset_command_timeout(self) -> None:
    """Sends a reset command timeout command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--reset-command-timeout')
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Reset_command_timeout'])

  def _enter_safe_start(self) -> None:
    """Sends an enter safe start command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--enter-safe-start')
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Enter_safe_start'])

  def _exit_safe_start(self) -> None:
    """Sends an exit safe start command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--exit-safe-start')
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Exit_safe_start'])

  def _deenergize(self) -> None:
    """Sends a deenergize command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--deenergize')
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Deenergize'])

  def _energize(self) -> None:
    """Sends an energize command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--energize')
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Energize'])

  def _set_step_mode(self) -> None:
    """Sends a set step mode command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--step-mode', str(self._step_mode))
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Set_step_mode'],
                        value=Tic_step_mode[self._step_mode])

  def _set_current_limit(self) -> None:
    """Sends a set current limit command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--current', str(self._current_limit))
    elif self._backend == 'USB':
      if self._model == '36v4':
        self._usb_command(request=Tic_cmd['Set_setting'],
                          value=self._unrestricted_current_limit,
                          index=Tic_settings['Unrestricted_current_limit'])
      self._usb_command(request=Tic_cmd['Set_current_limit'],
                        value=self._current_index)

  def _set_max_speed(self, speed: float) -> None:
    """Clamps the speed within the limits and sets it."""

    # The given speed may first need to be reduced or increased in order to
    # comply with the Tic ratings
    if abs(speed) > self._to_mm(Tic_max_speed / 10000):
      print(
        "Requested speed exceeding max allowed speed, setting to max "
        "allowed speed")
      max_speed = Tic_max_speed
    elif abs(speed) < self._to_mm(Tic_min_speed / 10000):
      print(
        "Requested speed below min possible speed, setting to min "
        "possible speed")
      max_speed = Tic_min_speed
    else:
      max_speed = abs(self._to_steps(speed * 10000))

    if self._backend == 'ticcmd':
      self._ticcmd('--max-speed', str(int(max_speed)))
    elif self._backend == 'USB':
      self._usb_32_bit(request=Tic_cmd['Set_max_speed'],
                       data=int(max_speed))

  def _set_max_accel(self) -> None:
    """Clamps the acceleration within the limits and sets it."""

    if self._backend == 'ticcmd':
      self._ticcmd('--max-accel', str(int(
        self._to_steps(self._max_accel * 100))))
    elif self._backend == 'USB':
      self._usb_32_bit(request=Tic_cmd['Set_max_accel'],
                       data=int(self._to_steps(self._max_accel * 100)))

  def _set_max_decel(self) -> None:
    """Clamps the deceleration within the limits and sets it."""

    if self._backend == 'ticcmd':
      self._ticcmd('--max-decel', str(int(
        self._to_steps(self._max_accel * 100))))
    elif self._backend == 'USB':
      self._usb_32_bit(request=Tic_cmd['Set_max_decel'],
                       data=int(self._to_steps(self._max_accel * 100)))

  def _set_position(self, position: float) -> None:
    """Sends a set position command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--position', str(int(self._to_steps(position))))
    elif self._backend == 'USB':
      self._usb_32_bit(request=Tic_cmd['Set_target_position'],
                       data=int(self._to_steps(position)))

  def _set_velocity(self, velocity: float) -> None:
    """Sends a set velocity command."""

    if self._backend == 'ticcmd':
      self._ticcmd('--velocity', str(int(velocity)))
    elif self._backend == 'USB':
      self._usb_32_bit(request=Tic_cmd['Set_target_velocity'],
                       data=int(velocity))

  def _get_max_speed(self) -> float:
    """Reads the maximum speed from the motor."""

    if self._backend == 'ticcmd':
      return self._to_mm(
        yaml.load(self._ticcmd('-s', '--full'), Loader=yaml.FullLoader)
        ['Max speed'] / 10000)
    elif self._backend == 'USB':
      return self._to_mm(int.from_bytes(
          self._usb_command(request_type=Tic_usb_request['Var'],
                            request=Tic_cmd['Get_variable'],
                            index=Tic_var['Max_speed'],
                            data_or_length=4),
          byteorder='little',
          signed=False) / 10000)

  def _set_pin_function(self, pin_func: Dict[str, str]) -> None:
    """Sets the pin function bitfields.

    Sends a command for setting each pin separately, and three commands for
    setting the Kill switch, Limit switch forward and Limit switch reverse
    bitfields.
    """

    if self._backend == 'ticcmd':
      pass
    elif self._backend == 'USB':
      # Reads the current bitfields
      kill_switch_map = int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_setting'],
                          index=Tic_settings['Kill_switch_map'],
                          data_or_length=1),
        byteorder='little',
        signed=False)
      limit_forward_map = int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_setting'],
                          index=Tic_settings['Limit_switch_forward_map'],
                          data_or_length=1),
        byteorder='little',
        signed=False)
      limit_reverse_map = int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_setting'],
                          index=Tic_settings['Limit_switch_reverse_map'],
                          data_or_length=1),
        byteorder='little',
        signed=False)
      for pin in pin_func:
        # Modifies the three bitfields
        if pin_func[pin] == 'Default':
          kill_switch_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_forward_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_reverse_map &= 0xFF - (1 << Tic_pins_bit[pin])
        elif pin_func[pin] == 'Kill switch':
          kill_switch_map |= 1 << Tic_pins_bit[pin]
          limit_forward_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_reverse_map &= 0xFF - (1 << Tic_pins_bit[pin])
        elif pin_func[pin] == 'Limit switch forward':
          kill_switch_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_forward_map |= 1 << Tic_pins_bit[pin]
          limit_reverse_map &= 0xFF - (1 << Tic_pins_bit[pin])
        elif pin_func[pin] == 'Limit switch reverse':
          kill_switch_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_forward_map &= 0xFF - (1 << Tic_pins_bit[pin])
          limit_reverse_map |= 1 << Tic_pins_bit[pin]

        # Sends an individual command for each pin
        self._usb_command(request=Tic_cmd['Set_setting'],
                          value=Tic_pin_modes[pin_func[pin]],
                          index=Tic_settings[pin + '_config'])
      # Sets the three bitfields
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=kill_switch_map,
                        index=Tic_settings['Kill_switch_map'])
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=limit_forward_map,
                        index=Tic_settings['Limit_switch_forward_map'])
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=limit_reverse_map,
                        index=Tic_settings['Limit_switch_reverse_map'])

  def _set_pin_polarity(self, pin_pol: Dict[str, str]) -> None:
    """Sets the switch polarity bitfield."""

    if self._backend == 'ticcmd':
      pass
    elif self._backend == 'USB':
      current = int.from_bytes(
        self._usb_command(request_type=Tic_usb_request['Var'],
                          request=Tic_cmd['Get_setting'],
                          index=Tic_settings['Switch_polarity_map'],
                          data_or_length=1),
        byteorder='little',
        signed=False)
      for pin in pin_pol:
        if Tic_pin_polarity[pin_pol[pin]]:
          current |= 1 << Tic_pins_bit[pin]
        else:
          current &= 0xFF - (1 << Tic_pins_bit[pin])
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=current,
                        index=Tic_settings['Switch_polarity_map'])

  def _reset(self) -> None:
    """Resets the Tic and reloads the settings."""

    if self._backend == 'ticcmd':
      pass
    elif self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Reset'])

  def _usb_command(self,
                   request_type: int = Tic_usb_request['Cmd'],
                   request: int = 0,
                   value: int = 0,
                   index: int = 0,
                   data_or_length: int = 0) -> Union[bytearray, int]:
    """Wrapper for sending a USB control transfer."""

    with self._lock:
      try:
        result = self._dev.ctrl_transfer(bmRequestType=request_type,
                                         bRequest=request,
                                         wValue=value,
                                         wIndex=index,
                                         data_or_wLength=data_or_length)
      except usb.core.USBError:
        raise IOError("An error occurred during USB communication")
    return result

  def _usb_32_bit(self, request: int, data: int) -> None:
    """Wrapper for sending USB requests containing 32-bits values."""

    value = data & 0xFFFF
    index = data >> 16 & 0xFFFF
    self._usb_command(request=request, value=value, index=index)

  def _ticcmd(self, *args: str) -> bytes:
    """Wrapper for calling ticcmd in a subprocess."""

    with self._lock:
      return subprocess.check_output(['ticcmd'] + ['-d'] +
                                     [self._serial_number]
                                     + list(args))

  def _thread_shutoff(self) -> None:
    """Thread for deenergizing the motor after a given period of inactivity.

    This thread reads the speed every 0.1s, and increments a timer if the speed
    is `0`. Once the timer reaches `_t_shutoff`, deenergizes the motor. The
    timer is reset if a speed or position command is issued, or if the speed is
    not `0`.
    """

    timer = 0
    while not self._close:
      time.sleep(0.01)

      while self._timer_shutoff:
        # Exit if close flag raised
        if self._close:
          break

        # Resetting timer if reset flag raised
        if self._reset_timer_shutoff:
          timer = 0
          self._reset_timer_shutoff = False

        # Checking if the motor is moving
        if self.get_speed() == 0:
          timer += 0.1
        else:
          timer = 0
        time.sleep(0.1)

        # Finally deenergizing the motor if all the conditions are met
        if timer > self._t_shutoff and \
                not self._reset_timer_shutoff and \
                not self._close:
          self._enter_safe_start()
          self._deenergize()
          self._timer_shutoff = False
          self._RCT = False  # Stopping the RCT thread as well
          timer = 0

  def _thread_rct(self) -> None:
    """Thread for sending the reset command timeout command every `0.5s`.

    This prevents the motor from stopping because of a reset command timeout
    error. Only sends the command when the motor is energized.
    """

    # Setting command timeout to 1000ms
    if self._backend == 'USB':
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=0xE8,
                        index=Tic_settings['Command_timeout_low'])
      self._usb_command(request=Tic_cmd['Set_setting'],
                        value=0x03,
                        index=Tic_settings['Command_timeout_high'])

    while not self._close:
      time.sleep(0.01)
      while self._RCT:
        if self._close:
          break
        self._reset_command_timeout()
        time.sleep(0.5)
