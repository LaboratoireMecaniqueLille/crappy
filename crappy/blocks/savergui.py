# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup SaverGUI SaverGUI
# @{

## @file savergui.py
# @brief Saves data in a file.
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

import os
import numpy as np
np.set_printoptions(threshold='nan', linewidth=500)

from .masterblock import MasterBlock
from ..tool import DataPicker

class SaverGUI(MasterBlock):
  """Saves data in a file"""

  def __init__(self):
    """
    Saver(log_file) Saves data in a file.

    Be aware that the log file needs to be cleaned before
    starting this function, otherwise it just keep writing a the end of the file.
    First line of the file will be meta-data. If file already exists, skips the
    meta-data writing.

    Args:
        log_file : string
            Path to the log file. If non-existant, will be created.
    """
    super(SaverGUI, self).__init__()

  # self.log_file=log_file
  # self.existing=False
  # if not os.path.exists(os.path.dirname(self.log_file)):
  # 	# check if the directory exists, otherwise create it
  # 	os.makedirs(os.path.dirname(self.log_file))
  # if os.path.isfile(self.log_file): # check if file exists
  # 	self.existing=True

  def main(self):
    first = True
    datapicker = DataPicker(self.inputs[1])
    try:
      while True:
        RecordOptions = self.inputs[0].recv()
        if RecordOptions['RecordFlag'] == 1:
          # Check of existing directory and files
          self.log_file = RecordOptions['RecordPath']
          self.existing = False
          if not os.path.exists(os.path.dirname(self.log_file)):
            # check if the directory exists, otherwise create it
            os.makedirs(os.path.dirname(self.log_file))
          if os.path.isfile(self.log_file):  # check if file exists
            self.existing = True
          Data = datapicker.get_data()
          data = Data.values()
          data = np.transpose(data)
          fo = open(self.log_file, "a")  # "a" for appending
          fo.seek(0, 2)  # place the "cursor" at the end of the file
          if first and not (self.existing):
            # legend_=Data.columns
            legend_ = Data.keys()
            fo.write(str([legend_[i] for i in range(len(legend_))]) + "\n")
            first = False
          data_to_save = str(data) + "\n"
          fo.write(data_to_save)
          fo.close()
      datapicker.close()
    except KeyboardInterrupt:
      print "KeyboardInterrupt received in SaverGUI"
      datapicker.close()
      # raise
    except Exception as e:
      print "Exception in SaverGUI %s: %s" % (os.getpid(), e)
      datapicker.close()
      # raise
      # time.sleep(1)
