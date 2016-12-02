# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Compacter Compacter
# @{

## @file _compacter.py
# @brief This block must be used to send data to the Saver or the Grapher.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _meta import MasterBlock
import os
from ..links._link import TimeoutError
from collections import OrderedDict


class Compacter(MasterBlock):
    """Many to one block. Compact several data streams into arrays."""

    def __init__(self, acquisition_step):
        """
        Read data inputs and save them in a panda dataframe of length acquisition_step.

        This block must be used to send data to the Saver or the Grapher.
        Input values sent by the Links must be array (1D).
        If you have multiple data input from several streamers, use multiple Compacter.
        You should use several input only if you know that they have the same frequency.
        You can have multiple outputs.

        Args:
            acquisition_step : int
                Number of values to save in each data-stream before returning the array.

        Returns:
            dict : OrderedDict(shape (number_of_values_in_input,acquisition_step))
        """
        super(Compacter, self).__init__()
        self.acquisition_step = acquisition_step

    def main(self):
        try:
            print "Compacter / Main loop: PID", os.getpid()
            while True:
                for i in xrange(self.acquisition_step):
                    if i == 0:
                        retrieved_data = self.inputs[0].recv()
                    else:
                        Data1 = self.inputs[0].recv()
                    if len(self.inputs) != 1:
                        for k in range(1, len(self.inputs)):
                            data_recv = self.inputs[k].recv()
                            if i == 0:
                                retrieved_data.update(data_recv)
                            else:
                                Data1.update(data_recv)
                    if i != 0:
                        try:
                            retrieved_data = OrderedDict(zip(retrieved_data.keys(), [retrieved_data.values()[t] + (Data1.values()[t],) for t in
                                                                 range(len(retrieved_data.keys()))]))
                        except TypeError:
                            retrieved_data = OrderedDict(zip(retrieved_data.keys(), [(retrieved_data.values()[t],) + (Data1.values()[t],) for t in
                                                                 range(len(retrieved_data.keys()))]))
                try:
                    for j in range(len(self.outputs)):
                        self.outputs[j].send(retrieved_data)
                    # print "compacted data : ",retrieved_data
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in Compacter %s: %s" % (os.getpid(), e)
            # raise
