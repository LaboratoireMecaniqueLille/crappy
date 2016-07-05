# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup ComediActuator ComediActuator
# @{

## @file _comediActuator.py
# @brief  Comedi actuator object, commands the output of comedi cards
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

from ._meta import command
import comedi as c


class ComediActuator(command.Command):
    """Comedi actuator object, commands the output of comedi cards"""

    def __init__(self, device='/dev/comedi0', subdevice=1, channel=0, range_num=0, gain=1, offset=0):
        ## @fn __init__()
        # @brief Convert wanted tension value into digital values and send it to the
        # output of some Comedi-controlled card.
        #
        # @param Output is (command * gain) + offset.
        #
        # @param device : str, default = '/dev/comedi0'
        #         Path to the device.
        # @param subdevice : int, default = 1
        #         Subdevice 1 is the output.
        # @param channel : int, default = 0
        #         The desired output channel.
        # @param range_num : int, default = 0
        #         See the comedi documentation for different values.
        # @param gain : float, default = 1
        #         Multiplication gain for the output.
        # @param offset : float, default = 0
        #         Add this value to your output.
        #
        super(ComediActuator, self).__init__()
        self.subdevice = subdevice
        self.channel = channel
        self.range_num = range_num
        self.gain = gain
        self.offset = offset
        self.device = c.comedi_open(device)
        self.new()

    def new(self):
        self.maxdata = c.comedi_get_maxdata(self.device, self.subdevice, self.channel)
        self.range_ds = c.comedi_get_range(self.device, self.subdevice, self.channel, self.range_num)
        c.comedi_dio_config(self.device, 2, self.channel, 1)

    def set_cmd(self, cmd):
        ## @fn set_cmd()
        # @brief Convert the tension value to a digital value and send it to the output.
        # @param cmd value to convert
        self.out = (cmd * self.gain) + self.offset
        out_a = c.comedi_from_phys(self.out, self.range_ds, self.maxdata)  # convert the cmd
        c.comedi_data_write(self.device, self.subdevice, self.channel, self.range_num, c.AREF_GROUND,
                            out_a)  # send the signal to the controler

    def on(self):
        c.comedi_dio_write(self.device, 2, self.channel, 1)

    def off(self):
        c.comedi_dio_write(self.device, 2, self.channel, 0)

    def close(self):
        ## @fn close()
        # @brief close the output.
        #
        ret = c.comedi_close(self.device)
        if ret != 0: raise Exception('comedi_close failed...')
