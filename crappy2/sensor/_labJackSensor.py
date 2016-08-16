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
from collections import OrderedDict
import Tkinter


class LabJackSensor(acquisition.Acquisition):
    """Sensor class for LabJack devices."""

    def __init__(self, mode="single", channels="AIN0", gain=1, offset=0, chan_range=10, resolution=0, scan_rate=100,
                 handle=None, identifier=None):
        """
        Convert tension value into digital values, on several channels.

        Available modes at the moment :

        - Single : Output is (measured_value * gain) + offset, can acquire at 1 kHz max.
        - Thermocouple : Output is a temperature in degree celsius.
        - Streamer : Output is (measured_value * gain) + offset, can acquire at 100 kHz max.

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

        self.chan_range = var_tester(chan_range, self.nb_channels)
        self.resolution = var_tester(resolution, self.nb_channels)

        self.gain = var_tester(gain, self.nb_channels)
        self.offset = var_tester(offset, self.nb_channels)

        self.mode = mode.lower()

        if self.mode == "streamer":
            self.a_scan_list = ljm.namesToAddresses(self.nb_channels, self.channels)[0]
            self.scan_rate = scan_rate
            self.scans_per_read = int(scan_rate / 10.)

        self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, str(identifier) if identifier else "ANY") \
            if not handle else handle  # open first found labjack OR labjack identified

        self.new()

    def new(self):
        """
        Initialise the device and create the handle
        """
        res_max = 12 if ljm.eReadName(self.handle, "WIFI_VERSION") > 0 else 8  # Test if LabJack is pro or not
        assert False not in [0 <= self.resolution[chan] <= res_max for chan in range(self.nb_channels)], \
            "Wrong definition of resolution index. INDEX_MAX for T7: 8, for T7PRO: 12"

        if self.mode == "single":

            suffixes = OrderedDict(("_RANGE", self.chan_range),
                                   ("_RESOLUTION_INDEX", self.resolution),
                                   ("_EF_INDEX", 1),                        # for applying a slope and offset
                                   ("_EF_CONFIG_D", self.gain),             # index to set the gain
                                   ("_EF_CONFIG_E", self.offset))           # index to set the offset

        elif self.mode == "thermocouple":
            suffixes = OrderedDict(("_EF_INDEX", 22),                       # for thermocouple measures
                                   ("_EF_CONFIG_A", 1),                     # for degrees C
                                   ("_EF_CONFIG_B", 60052))                 # for type K

        elif self.mode == "streamer":
            a_names = ["AIN_ALL_RANGE", "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
            a_values = [int(self.chan_range[0]), 0, int(self.resolution[0])]
      #     a_names = ["AIN_ALL_NEGATIVE_CH", "AIN_ALL_RANGE", "STREAM_SETTLING_US",
      #                 "STREAM_RESOLUTION_INDEX"]
      #     a_values = [ljm.constants.GND, self.chan_range, 0, 0]

        else:
            raise Exception("Unrecognized mode. Check documentation.")

        if self.mode == "single" or self.mode == "thermocouple":
            a_values = []
            a_names = []
            for chan_iteration in range(self.nb_channels):
                for count, key in enumerate(suffixes):
                    a_names.append(self.channels[chan_iteration] + suffixes.keys()[count])
                    if isinstance(suffixes.get(key), list):
                        a_values.append(suffixes.get(key)[chan_iteration])
                    else:
                        a_values.append(suffixes.get(key))
        ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)

    def start_stream(self):
        # self.root = Tkinter.Tk()
        # self.rows = ['Chronometer', 'Device Buffer', 'Software Buffer']
        #
        # self.root.title('LabJack Streamer')
        #
        # for row_index, row in enumerate(self.rows):
        #     Tkinter.Label(self.root, text=row, borderwidth=10).grid(row=row_index, column=0)
        #
        # self.root.resizable(width=False, height=False)
        # self.root.mainloop()
        ljm.eStreamStart(self.handle, self.scans_per_read, self.nb_channels, self.a_scan_list, self.scan_rate)

    def get_data(self, mock=None):
        """
        Read the signal on all pre-defined channels, one by one.
            \todo
              Make the acquisition more efficient. Add an option to stream more than 1 channel.
        """
        try:
            if not self.mode == "streamer":
                results = ljm.eReadNames(self.handle, self.nb_channels, self.channels_index_read)
                return time.time(), results
            else:
                retrieved_from_buffer = ljm.eStreamRead(self.handle)
                results = retrieved_from_buffer[0]
                timer = time.time()
                # Tkinter.Label(self.root, text=timer, borderwidth=10).grid(row=0, column=1)
                # Tkinter.Label(self.root, text=retrieved_from_buffer[1], borderwidth=10).grid(row=1, column=1)
                # Tkinter.Label(self.root, text=retrieved_from_buffer[2], borderwidth=10).grid(row=2, column=1)
                # self.root.update()
                print "Left on Device Buffer: %i " % retrieved_from_buffer[1]
                print "Left on LJM Buffer: %i " % retrieved_from_buffer[2]
                return timer, results
        except KeyboardInterrupt:
            self.close()
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
