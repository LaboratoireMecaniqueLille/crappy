# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjacksensor LabJackSensor
# @{

## @file _labJackSensor.py
# @brief  Sensor class for LabJack devices.
# @author Francois Bari
# @version 0.9
# @date 18/08/2016

from labjack import ljm
from time import time
from sys import exc_info
from ._meta import acquisition
from collections import OrderedDict
from multiprocessing import Process, Queue
from Tkinter import Tk, Label


class LabJackSensor(acquisition.Acquisition):
    """Sensor class for LabJack devices."""

    def __init__(self, mode="single", channels="AIN0", gain=1, offset=0, chan_range=10, resolution=0, scan_rate=100,
                 scans_per_read=None, handle=None, identifier=None):
        """
        Convert tension value into digital values, on several channels, using LabJack Devices.

        Args:

            mode :         str.
                           Available modes at the moment :
                           - Single : Output is (measured_value * gain) + offset, can acquire at 1 kHz max.
                           - Thermocouple : Output is a temperature in degree celsius.
                           - Streamer : Output is measured_value, can acquire at 100 kHz max.

            channels :     int, str or list of int or str, default = 0
                           The desired input channel(s). If int, will be assumed as "AIN".

            gain :         float or list of float, default = 1
                           Multiplication gain for each channel. If there is multiple channels
                           for a single gain, it will be applied to all.

            offset :       float, default = 0
                           Add this value for each channel. If there is multiple channels
                           for a single offset, it will be applied to all.

            chan_range :   int or float, default = 10. Can be 10, 1, 0.1 or 0.01, depending on the voltage range
                           to measure. Put the absolute maximum of your expected values.

            resolution :   int, resolution index for each channel (T7 : 0 to 8, T7-PRO : 0 to 12) ~11 to 22 bits
                           depending on the device, the chan_range and the resolution index.
                           higher resolution index = higher resolution, but higher latency.

            scan_rate :    STREAMER MODE ONLY : int, defines how many scans to perform during streaming.
                           Effective acquisition frequency per channel = scan_rate/nb_channels

            scans_per_read:STREAMER MODE ONLY : int, defines how many samples to collect during one loop.
                           If undefined, will be assumed as a fraction of scan_rate, determined for performance.

            handle:        Used if using labjack as I/O device at the same time. Unused for the moment (18/08/2016)

            identifier:    str. Used if multiple labjacks are connected. The identifier could be anything
                           that could define the device : serial number, name, wifi version..
            """
        super(LabJackSensor, self).__init__()

        def var_tester(var, nb_channels):
            """Used to check if the user entered correct parameters."""
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
            """Additionnal variables for streamer function only."""
            self.a_scan_list = ljm.namesToAddresses(self.nb_channels, self.channels)[0]
            self.scan_rate = scan_rate
            self.scans_per_read = scans_per_read if not None else int(self.scan_rate / 5.)
            global queue
            queue = Queue()
        self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, str(identifier) if identifier else "ANY") \
            if not handle else handle  # open first found labjack OR identified labjack

        self.new()

    class DialogBox:
        """Dialog box that pops when using streamer function."""
        def __init__(self, scan_rate, scans_per_sample):
            self.root = Tk()
            self.root.title('LabJack Streamer Information')
            self.root.resizable(width=False, height=False)

            self.first_column = \
                ['Set Scan Rate', 'Samples Collecting Rate', 'Chronometer', 'Device Buffer', 'Software Buffer']
            self.second_column = \
                ['%.1f kHz' % (scan_rate / 1000), '%.1f kScans' % (scans_per_sample / 1000), 0.0, 0, 0]
            for row_index, first_column in enumerate(self.first_column):
                Label(self.root, text=first_column, borderwidth=10).grid(row=row_index, column=0)
                Label(self.root, text=self.second_column[row_index], borderwidth=10).grid(row=row_index, column=1)
            self.update()

        def update(self):
            """Method to update data inside the dialog box. The window is updated every time data in queue occurs."""
            try:
                global queue
                array = queue.get()
                t0 = array[0]
                while True:
                    array[0] = '%.1f' % (array[0] - t0)
                    for row_index, value in enumerate(array):
                        Label(self.root, text=value, borderwidth=10).grid(row=row_index + 2, column=1)
                    self.root.update()
                    array = queue.get()
            except KeyboardInterrupt:
                pass
            except Exception:
                raise

    def new(self):
        """
        Initialize the device.
        """
        try:
            res_max = 12 if ljm.eReadName(self.handle, "WIFI_VERSION") > 0 else 8  # Test if LabJack is pro or not
            assert False not in [0 <= self.resolution[chan] <= res_max for chan in range(self.nb_channels)], \
                "Wrong definition of resolution index. INDEX_MAX for T7: 8, for T7PRO: 12"

            if self.mode == "single":
                to_write = OrderedDict([
                                       ("_RANGE", self.chan_range),
                                       ("_RESOLUTION_INDEX", self.resolution),
                                       ("_EF_INDEX", 1),                        # for applying a slope and offset
                                       ("_EF_CONFIG_D", self.gain),             # index to set the gain
                                       ("_EF_CONFIG_E", self.offset)            # index to set the offset
                                       ])

            elif self.mode == "thermocouple":
                to_write = OrderedDict([
                                       ("_EF_INDEX", 22),                       # for thermocouple measures
                                       ("_EF_CONFIG_A", 1),                     # for degrees C
                                       ("_EF_CONFIG_B", 60052)                  # for type K
                                       ])

            elif self.mode == "streamer":
                a_names = ["AIN_ALL_RANGE", "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
                a_values = [int(self.chan_range[0]), 0, int(self.resolution[0])]

            else:
                raise Exception("Unrecognized mode. Check documentation.")

            if self.mode == "single" or self.mode == "thermocouple":
                a_values = []
                a_names = []
                for chan_iteration in range(self.nb_channels):
                    for count, key in enumerate(to_write):
                        a_names.append(self.channels[chan_iteration] + to_write.keys()[count])
                        if isinstance(to_write.get(key), list):
                            a_values.append(to_write.get(key)[chan_iteration])
                        else:
                            a_values.append(to_write.get(key))
            ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)
        except:
            self.close()
            raise

    def start_stream(self):
        """Method to initialize a streaming data."""
        try:
            ljm.eStreamStart(self.handle, self.scans_per_read, self.nb_channels, self.a_scan_list, self.scan_rate)
            Process(target=self.DialogBox, args=(self.scan_rate, self.scans_per_read)).start()

        except KeyboardInterrupt:
            self.close()
            pass
        except Exception:
            raise

    def get_data(self, mock=None):
        """
        Read the signal on all pre-defined channels, one by one.
        """
        try:
            results = ljm.eReadNames(self.handle, self.nb_channels, self.channels_index_read)
            return time(), results

        except KeyboardInterrupt:
            self.close()
            pass

        except Exception:
            print(exc_info()[1])
            self.close()
            raise

    def get_stream(self):
        """
        Read the device buffer if scan_mode is set.
        """
        global queue
        try:
            retrieved_from_buffer = ljm.eStreamRead(self.handle)
            results = retrieved_from_buffer[0]
            timer = time()
            queue.put([timer, retrieved_from_buffer[1], retrieved_from_buffer[2]])
            return timer, results

        except KeyboardInterrupt:
            ljm.eStreamStop(self.handle)
            self.close()
            pass

        except Exception:
            print(exc_info()[1])
            ljm.eStreamStop(self.handle)
            self.close()
            raise

    def close(self):
        """
        Close the device.
        """
        if ljm.eReadName(self.handle, "STREAM_ENABLE"):
            ljm.eStreamStop(self.handle)
        ljm.close(self.handle)
        print "LabJack device closed"
        # try:
        #     ljm.eStreamStop(self.handle)
        # except ljm.LJMError:  # if no streamer open
        #     pass
        # finally:
        #     ljm.close(self.handle)
        #     print ("LabJack device closed")
