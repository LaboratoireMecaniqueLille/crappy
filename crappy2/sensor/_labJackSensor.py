# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjacksensor LabJackSensor
# @{

## @file _labJackSensor.py
## @brief  Sensor class for LabJack devices.
## @author Robin Siemiatkowski
## @version 0.1
## @date 21/06/2016

from labjack import ljm
import time
import sys
from collections import OrderedDict
from ._meta import acquisition
from .._warnings import deprecated as deprecated


class LabJackSensor(acquisition.Acquisition):
    """Sensor class for LabJack devices."""

    def __init__(self, channels=[0], chan_range=10, gain=1, offset=0, resolution=0, mode="single", scanRate=100,
                 scansPerRead=1):
        """
        Convert tension value into digital values, on several channels.

        Output is (measured_value * gain) + offset.

        Parameters
        ----------
        channel : int/str or list of int/str, default = 0
        The desired input channel(s). If int, will be assumed as "AIN".
        chan_range : int or float, default = 10
        Put the absolute maximum of your expected values.
        gain : float or list of float, default = 1
        Multiplication gain for each channel. If there is multiple channels
        for a single gain, it will be applied to all.
        offset : float, default = 0
        Add this value for each channel. If there is multiple channels
        for a single offset, it will be applied to all.
        mode : {"single","streamer"}
        Whether you want to stream value or read value by value. Be aware that the
        streamer will send values in the LabJack's buffer. Once full, the LabJack
        will raise an Exception and stop working. You need to keep reading the 
        buffer in order to keep the streamer working.
        scanRate : int
        For "streamer" mode only. Define the reading frequency.
        scanPerRead : int
        For "streamer" mode only. Define how many values you want to read at each
        scans.
        """
        super(LabJackSensor, self).__init__()
        self.channels = list(channels)
        self.chan_range = chan_range
        self.gain = gain
        self.offset = offset
        self.mode = mode
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.resolution = resolution

        if type(self.channels) == str or len(self.channels) == 1:  # for get_data
            self.nchans = 1
        if type(self.channels) == list:  # if multiple channels
            self.nchans = len(self.channels)
            # self.channels=["AIN"+str(chan) for chan in self.channels if type(chan)!=str else chan]
            self.channels = ["AIN" + str(chan) if type(chan) != str else chan for chan in self.channels]
            if type(self.gain) == int:
                self.gain = [self.gain] * self.nchans
            if type(self.offset) == int:
                self.offset = [self.offset] * self.nchans
            if type(self.chan_range) == int or type(self.chan_range) == float:
                self.chan_range = [self.chan_range] * self.nchans
            if mode == "single":  # if mode "streamer", the new() must be called in the right process
                self.new()
        else:
            raise Exception("channels must be int or list")

    def new(self):
        """
        Initialise the device and create the handle
        """
        self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, "ANY")
        if self.mode == "streamer":
            # self.channels = ["AIN0", "AIN1", "AIN2", "AIN3"] #Scan list names to stream
            numAddresses = len(self.channels)
            aScanList = ljm.namesToAddresses(numAddresses, self.channels)[0]
            aName_prefix = ["AIN_ALL_NEGATIVE_CH", "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
            suffixes = ["_RANGE"]
            aNames = aName_prefix + [chan + s for chan in self.channels for s in suffixes]
            temp_values = [[self.chan_range[chan]] for chan, a in enumerate(self.channels)]
            aValues = [ljm.constants.GND, 0, self.resolution] + [item for sublist in temp_values for item in sublist]
            # print aValues
            # print "----------------------"
            # print aNames
            ljm.eWriteNames(self.handle, len(aNames), aNames, aValues)

            # Configure and start stream
            scanRate = ljm.eStreamStart(self.handle, self.scansPerRead, numAddresses, aScanList, self.scanRate)

        elif self.mode == "single":
            numAddresses = len(self.channels)
            # suffixes = ["_EF_INDEX","_NEGATIVE_CH", "_RANGE", "_RESOLUTION_INDEX", "_EF_CONFIG_D", "_EF_CONFIG_E"] # conf for automatic gain/offset
            suffixes = ["_NEGATIVE_CH", "_RANGE", "_RESOLUTION_INDEX"]
            aNames = [chan + s for chan in self.channels for s in suffixes]
            # aValues = [[1,ljm.constants.GND, self.chan_range[chan],0,self.gain[chan],self.offset[chan]] for chan,a in enumerate(self.channels)]
            aValues = [[ljm.constants.GND, self.chan_range[chan], self.resolution] for chan, _ in
                       enumerate(self.channels)]
            aValues = [item for sublist in aValues for item in sublist]  # flatten
            # print aValues
            # print "----------------------"
            # print aNames
            ljm.eWriteNames(self.handle, len(aNames), aNames, aValues)

        else:
            raise Exception("Invalid mode, please select 'single' or 'streamer'")

    def read_stream(self):
        """
        Read the signal on all pre-definied channels at the same time.
        """
        try:
            # print "here1"
            ret = ljm.eStreamRead(self.handle)
            # print ret
            data = ret[0]
            # print "stream1", data
            try:
                # print "data : ",data
                # if self.scansPerRead!=1:
                # data=zip(*[iter(data)]*self.scansPerRead)
                curSkip = data.pop(-9999.0)  # delete non-valid values
                print "skipped some data!"
            except TypeError:  # No corrupted data
                data = [a * b + c for a, b, c in
                        zip(data, self.gain * self.scansPerRead, self.offset * self.scansPerRead)]  # apply coefficients
                # ret=OrderedDict(zip(['t(s)'],[time.time()]))
                # ret.update(OrderedDict(zip(self.channels,data)))
                # results=[a*b+c for a,b,c in zip(results,self.gain,self.offset)]
                # ret=OrderedDict(zip(['t(s)'],[time.time()]))
                # ret.update(OrderedDict(zip(channels,results)))
                # data.insert(0,)
                # print "data2 : ",data
                return time.time(), data
        except ljm.LJMError:
            # print "error 1"
            ljme = sys.exc_info()[1]
            print(ljme)
            self.close()
            raise Exception("Error in LabJack Streamer")
        except Exception:
            # print "exception in streamer"
            e = sys.exc_info()[1]
            print(e)
            self.close()
            raise

    def get_data(self, mock=None):
        """
        Read the signal on all pre-definid channels, one by one.
        """
        try:
            # numFrames = len(channels)
            # names = ["AIN"+str(chan) if type(chan)!=str else chan for chan in channels]
            # results = ljm.eReadNames(self.handle, numFrames, names)
            results = ljm.eReadNames(self.handle, self.nchans, self.channels)
            results = [a * b + c for a, b, c in zip(results, self.gain, self.offset)]
            # ret=OrderedDict(zip(['t(s)'],[time.time()]))
            # ret.update(OrderedDict(zip(channels,results)))
            # results.insert(0,)
            return time.time(), results
        except KeyboardInterrupt:
            self.close()
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
