# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Reader Reader
# @{

## @file _reader.py
# @brief Children class of MasterBlock. Read and print the input Link.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

from _meta import MasterBlock
import time
import os


class Reader(MasterBlock):
    """
    Children class of MasterBlock. Read and print the input Link.
    """

    def __init__(self, k):
        """
        Create a reader that prints k and the input data in continuous.

        Args:
            k : printable
                Some identifier for this particular instance of Reader
        """
        super(Reader, self).__init__()
        self.k = k

    def main(self):
        try:
            while True:
                for input_ in self.inputs:
                    self.data = input_.recv()
                    print "Received by {0}: {1}.".format(input_.name, self.data)
        except Exception as e:
            print "Exception in reader (pid:{0}): {1}".format(os.getpid(), e)
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected exception."
