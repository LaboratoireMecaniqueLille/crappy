# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup DaqmxActuator DaqmxActuator
# @{

## @file _comediActuator.py
# @brief  Actuator class for Daqmx devices.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

try:
    from PyDAQmx import *
except ImportError:
    print "WARNING : no module PyDAQmx installed, sensor won't work!"
import numpy
import time
from ._meta import command


def get_daqmx_devices_names():
    buffer_size = 4096
    buffer = ctypes.create_string_buffer(buffer_size)
    DAQmxGetSysDevNames(buffer, buffer_size)
    print len(buffer.value.split(",")), " devices detected: ", buffer.value
    return buffer.value.split(",")


class DaqmxActuator(command.Command):
    """Actuator class for Daqmx devices."""
    """PXI1Slot2, PXI1Slot3"""

    def __init__(self, device='Dev1', channels=0, range_num=0, mode="single"):
        ## @fn __init__()
        # @brief Convert wanted tension value into digital values and send it to the
        # output of some Daqmx-controlled card.
        #
        # @param device : str, default = 'dev1'
        #         Device name.
        # @param subdevice : int, default = 1
        #         Subdevice 1 is the output.
        # @param channel : int, default = 0
        #         The desired output channel, index of the _channels_tab attribute (from 0 to 15).
        #             ["ai0", "ai1", "ai2", "ai3", "ai4", "ai5", "ai6", "ai7", "ai8", "ai9", "ai10", "ai11",
        #              "ai12", "ai13", "ai14", "ai15"]
        # @param range_num : int, default = 0
        #         Desired range, index of the _range_tab attribute (from 0 to 11):
        #             [[0.0, 0.5], [0.0, 1.0], [0.0, 2.5], [0.0, 5.0], [0.0, 7.5], [0.0, 10.0],
        #              [-0.5, 0.5], [-1.0, 1.0], [-2.5, 2.5], [-5.0, 5.5], [-7.5, 7.5], [-10.0, 10.0]]
        # @param mode: not used yet.
        #
        super(DaqmxActuator, self).__init__()
        self._ranges_tab = [[0.0, 0.5], [0.0, 1.0], [0.0, 2.5], [0.0, 5.0], [0.0, 7.5], [0.0, 10.0],
                            [-0.5, 0.5], [-1.0, 1.0], [-2.5, 2.5], [-5.0, 5.5], [-7.5, 7.5], [-10.0, 10.0]]
        self._channels_tab = ["ai0", "ai1", "ai2", "ai3", "ai4", "ai5", "ai6", "ai7", "ai8", "ai9", "ai10", "ai11",
                              "ai12", "ai13", "ai14", "ai15"]

        # Declaration of variable passed by reference
        self.mode = mode
        self.channels = channels
        self.range_num = range_num

        # if type(self.channels)==list:
        # if multiple channels
        # self.nchans=len(self.channels)
        # self.range_num=[self.range_num]*self.nchans
        # if type(self.gain)==int:
        # self.gain=[self.gain]*self.nchans
        # if type(self.offset)==int:
        # self.offset=[self.offset]*self.nchans
        # self.new()
        # else:
        # raise Exception("channels must be int or list")

        self.channel = self._channels_tab[channels]
        self.device = "%s/%s" % (device, self.channel)

    def new(self):
        DAQmxResetDevice(device)

        self.taskHandle = TaskHandle()
        self.read = int32()

        # DAQmx Configure Code
        DAQmxCreateTask("", byref(self.taskHandle))

        print self.device
        print type(self.device)

        buffer_size = 4096
        buffer = ctypes.create_string_buffer(buffer_size)
        DAQmxGetDeviceAttribute(device, DAQmx_Dev_ProductType, buffer)
        print buffer.value
        # if buffer.value == "PXIe-4331":
        #     print device
        #     DAQmxCreateAIBridgeChan(self.taskHandle, self.device,"",0.0,100.0,DAQmx_Val_VoltsPerVolt,
        #                             DAQmx_Val_FullBridge,DAQmx_Val_Internal,1,20, None)
        # else:
        DAQmxCreateAIVoltageChan(self.taskHandle, self.device, "", DAQmx_Val_Cfg_Default,
                                 self._ranges_tab[self.range_num][0], self._ranges_tab[self.range_num][1],
                                 DAQmx_Val_Volts, None)

    def set_cmd(self, cmd):
        ## @fn  set_cmd()
        # @brief write signal to the output.
        # TODO
        pass

    def close(self):
        ## @fn close()
        # @brief Close the connection
        if self.taskHandle:
            # DAQmx Stop Code
            DAQmxStopTask(self.taskHandle)
            DAQmxClearTask(self.taskHandle)
        else:
            raise Exception('closing failed...')

    def __str__(self):
        ## @fn __str__()
        # @brief prints out attributes values.
        return " Device: {0} \n Channels({1}): {2} \n Range({3}): min:{4} max: {5} \
               \n Mode: {6}".format(self.device,
                                    self.channels, self._channels_tab[self.channels],
                                    self.range_num, self._ranges_tab[self.range_num][0],
                                    self._ranges_tab[self.range_num][1], self.mode)
