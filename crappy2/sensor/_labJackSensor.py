# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjacksensor LabJackSensor
# @{

## @file _labJackSensor.py
# @brief  Sensor class for LabJack devices.
# @author Francois Bari
# @version 0.2
# @date 08/07/2016


from labjack import ljm
import time
import sys
from ._meta import acquisition
import collections


class LabJackSensor(acquisition.Acquisition):
    """Sensor class for LabJack devices."""

    def __init__(self, mode="single", channels="AIN0", gain=1, offset=0, chan_range=10, resolution=0, handle=None):
        """
        Convert tension value into digital values, on several channels.

        Available modes at the moment :

        - Single : Output is (measured_value * gain) + offset.
        - Thermocouple : Output is a temperature in degree celsius.

        Args:
            channels : int/str or list of int/str, default = 0
                        The desired input channel(s). If int, will be assumed as "AIN".
            chan_range : int or float, default = 10. Can be 10, 1, 0.1 or 0.01,
                        depending on the range of the voltage to measure. Put the absolute maximum of your expected values.
            resolution : int, resolution index for each channel (T7 : 0 to 8, T7-PRO : 0 to 12)
            gain : float, or list of float, default = 1
                    Multiplication gain for each channel. If there is multiple channels
                    for a single gain, it will be applied to all.
            offset : float, default = 0
                    Add this value for each channel. If there is multiple channels
                    for a single offset, it will be applied to all.
            """
        super(LabJackSensor, self).__init__()

        def var_tester(var, nb_channels):
            if isinstance(var, (int, float)):
                var = [var] * nb_channels
            elif isinstance(var, list) and len(var) == nb_channels and False not in [isinstance(var[i], (int, float))
                                                                                     for i in range(nb_channels)]:
                pass
            else:
                raise Exception(str(var) + " LabJack error: Wrong parameter definition. "
                                           "Parameters should be either int, float or a list of such")
            return var

        if not isinstance(channels, list):
            self.channels = [channels]
        else:
            self.channels = channels

        self.channels = ["AIN" + str(chan) if type(chan) is not str else chan for chan in self.channels]
        self.nb_channels = len(self.channels)
        self.channels_index_read = [self.channels[chan] + "_EF_READ_A" for chan in range(self.nb_channels)]

        self.gain = var_tester(gain, self.nb_channels)
        self.offset = var_tester(offset, self.nb_channels)
        self.chan_range = var_tester(chan_range, self.nb_channels)
        self.resolution = var_tester(resolution, self.nb_channels)

        self.mode = mode.lower()

        if not handle:
            self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, "ANY")  # open first found labjack
        else:
            self.handle = handle
        self.new()

    def new(self):
        """
        Initialise the device and create the handle
        """
        if ljm.eReadName(self.handle,
                         "WIFI_VERSION") > 0:  # Test if LabJack is a pro or not. Then verifies index values.
            res_max = 12
        else:
            res_max = 8

        assert False not in [0 <= self.resolution[chan] <= res_max for chan in range(self.nb_channels)], \
            "Wrong definition of resolution index. INDEX_MAX for T7: 8, for T7PRO: 12"

        aValues = []
        aNames = []

        if self.mode == "single":
            suffixes = (("_RANGE", self.chan_range),
                        ("_RESOLUTION_INDEX", self.resolution),
                        ("_EF_INDEX", 1),
                        ("_EF_CONFIG_D", self.gain),
                        ("_EF_CONFIG_E", self.offset))

        elif self.mode == "thermocouple":
            suffixes = (("_EF_INDEX", 22),  # for thermocouple measures
                        ("_EF_CONFIG_A", 1),  # for degrees C
                        ("_EF_CONFIG_B", 60052))  # for type K

        else:
            raise Exception("Unrecognized mode. Check documentation for details.")

        suffixes = collections.OrderedDict(suffixes)

        for chan in range(self.nb_channels):
            for count, key in enumerate(suffixes):
                aNames.append(self.channels[chan] + suffixes.keys()[count])
                if isinstance(suffixes.get(key), list):
                    aValues.append(suffixes.get(key)[chan])
                else:
                    aValues.append(suffixes.get(key))

        ljm.eWriteNames(self.handle, len(aNames), aNames, aValues)

    def get_data(self, mock=None):
        """
        Read the signal on all pre-defined channels, one by one.
        """
        try:
            results = ljm.eReadNames(self.handle, self.nb_channels, self.channels_index_read)
            return time.time(), results
        except KeyboardInterrupt:
            # self.close()
            pass
        except Exception:
            print(sys.exc_info()[1])
            self.close()
            raise

    def close(self):
        """
        Close the device.
        """
        try:
            ljm.eStreamStop(self.handle)
        except ljm.LJMError:  # if no streamer open
            pass
        finally:
            ljm.close(self.handle)
            print ("LabJack device closed")
