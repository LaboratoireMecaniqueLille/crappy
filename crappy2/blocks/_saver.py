# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Saver Saver
# @{

## @file _saver.py
# @brief Saves data in a file.
# @author Robin Siemiatkowski, Francois Bari
# @version 0.2
# @date 13/08/2016

from _meta import MasterBlock
import os
import numpy as np
import time

np.set_printoptions(threshold='nan', linewidth=500)


class Saver(MasterBlock):
    """Saves data in a file
    \todo
      Add an option to create a timestamp, instead of append only one file.
    """

    def __init__(self, log_file, *args, **kwargs):
        """
        Constructor of Saver class.

        Be aware that the log file needs to be cleaned before starting this function,
        otherwise it just keep writing a the end of the file.
        First line of the file will be meta-data. If file already exists, skips the
        meta-data writing.

        If the folder doesn't exists, creates it.

        Args:
            log_file : str
            Path to the log file. If non-existant, will be created.
            output_format : format of file created. Possible values : '.txt' (default), '.csv'
        """
        super(Saver, self).__init__()
        print "saver! : ", os.getpid()
        self.log_file = log_file
        self.existing = False
        self.output_format = log_file[-3:]
        if not os.path.exists(os.path.dirname(self.log_file)):
            # check if the directory exists, otherwise create it
            os.makedirs(os.path.dirname(self.log_file))
        if os.path.isfile(self.log_file):  # check if file exists
            self.existing = True

    def main(self):
        first = True
        while True:
            try:
                Data = self.inputs[0].recv()  # recv data
                data = Data.values()
                data = np.transpose(data)
                fo = open(self.log_file, "a", buffering=3)  # "a" for appending, buffering to limit r/w disk access.
                fo.seek(0, 2)  # place the "cursor" at the end of the file

                if first and not self.existing:
                    # legend_=Data.columns
                    legend_ = Data.keys()
                    if self.output_format == 'txt':
                        fo.write(str([legend_[i] for i in range(len(legend_))]) + "\n")
                    elif self.output_format == 'csv':
                        #  Write the first line : value, value,
                        for index in xrange(len(legend_) - 1):
                            fo.write(str(legend_[index]) + ',')
                        # Write the last line : value \n
                        fo.write(str(legend_[-1]) + '\n')
                    first = False

                if self.output_format == 'txt':
                    data_to_save = str(data) + "\n"
                    fo.write(data_to_save)
                    fo.close()

                elif self.output_format == 'csv':
                    print 'np shape data:', np.shape(data)
                    for column in xrange(np.shape(data)[0]):
                        for line in xrange(np.shape(data)[1] - 1):
                            fo.write(str(data[column][line]) + ',')
                        fo.write(str(data[column][-1]) + '\n')


            except KeyboardInterrupt:
                print 'KeyboardInterrupt received in saver'
                break
            except Exception as e:
                print "Exception in saver %s: %s" % (os.getpid(), e)
                # raise
