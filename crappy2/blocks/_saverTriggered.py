# coding: utf-8
from _meta import MasterBlock
import os
import numpy as np
import time

np.set_printoptions(threshold='nan', linewidth=500)

class SaverTriggered(MasterBlock):
    """Saves data in a file"""
    
    def __init__(self, log_file):
	
        super(SaverTriggered, self).__init__()
        print "saver! : ", os.getpid()
        self.log_file = log_file
        self.existing = False
        if not os.path.exists(os.path.dirname(self.log_file)):
            # check if the directory exists, otherwise create it
            os.makedirs(os.path.dirname(self.log_file))
        if os.path.isfile(self.log_file):  # check if file exists
            self.existing = True
    
    def main(self):
        first = True
        while True:
            try:
                # data=self.inputs[0].recv()
                if self.inputs[1].recv(blocking = False) != None:
		    self.log_file = self.inputs[1].recv()
		    print self.log_file
		    if not os.path.exists(os.path.dirname(self.log_file)):
			# check if the directory exists, otherwise create it
			os.makedirs(os.path.dirname(self.log_file))
		    if os.path.isfile(self.log_file):  # check if file exists
			self.existing = True
		    first = True
                Data = self.inputs[0].recv()  # recv data
                data = Data.values()
                data = np.transpose(data)
                fo = open(self.log_file, "a")  # "a" for appending
                fo.seek(0, 2)  # place the "cursor" at the end of the file
                if first and not self.existing:
                    # legend_=Data.columns
                    legend_ = Data.keys()
                    fo.write(str([legend_[i] for i in range(len(legend_))]) + "\n")
                    first = False
                data_to_save = str(data) + "\n"
                fo.write(data_to_save)
                fo.close()
            except KeyboardInterrupt:
                print 'KeyboardInterrupt received in saver'
                break
            except Exception as e:
                print "Exception in saver %s: %s" % (os.getpid(), e)
                # raise    
    