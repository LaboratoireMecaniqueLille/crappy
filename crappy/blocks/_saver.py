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

from _masterblock import MasterBlock
import os
import numpy as np
import datetime

np.set_printoptions(threshold='nan', linewidth=500)


class Saver(MasterBlock):
  """
  Saves data in a file.
  """

  def __init__(self, log_file, *args, **kwargs):
    """
    Constructor of Saver class.

    Be aware that if you doesn't use a stamp, the log file needs to be cleaned before starting this function,
    otherwise it just keep writing a the end of the file.
    First line of the file will be meta-data. If file already exists, skips the
    meta-data writing.

    If the folder doesn't exists, creates it.

    Args:
        log_file : str
        Path to the log file. If non-existant, will be created.
        output_format : format of file created. Possible values : '.txt' (default), '.csv'. If you want a new
        format, add it to this block then add to the known_formats list.
        stamp : adds a stamp to the file. Possible values are :
                - 'date': adds a date to the output file.
                    Example, if log_file = toto.csv : toto_2016-10-19_10:11.csv
                - anything else, other than False: adds a number to the output file, corresponding on how many
                files are present in the saving folder.
                    Example, if log_file = toto.csv, and folder contains 2 files with toto in it: toto_3.csv
    """
    super(Saver, self).__init__()
    self.stamp = kwargs.get('stamp', False)
    self.output_format = '.' + log_file.rsplit('.', 1)[
      -1]  # To define if it's a CSV, or a simple TXT where lists are saved directly.
    self.log_name = log_file.rstrip(self.output_format).rsplit('/', 1)[-1]
    self.known_formats = ['.txt', '.csv']
    assert self.output_format in self.known_formats, (
      'Unknown output format. Possible formats are:', self.known_formats)

    self.existing = False
    if not os.path.exists(os.path.dirname(log_file)):
      # check if the directory exists, otherwise create it
      os.makedirs(os.path.dirname(log_file))
    if os.path.isfile(log_file):  # check if file exists
      self.existing = True

    if self.stamp:
      if self.stamp == 'date':
        self.stamp_string = '{:%Y-%m-%d_%H:%M}'.format(datetime.datetime.now())
      else:
        self.nb_files_folder = os.listdir(os.path.dirname(log_file))
        self.nb_identical_files = [value.count(self.log_name) for _, value in enumerate(self.nb_files_folder)]
        self.stamp_string = str(self.nb_identical_files.count(1) + 1)
      self.log_file = log_file.rstrip(self.output_format) + '_' + self.stamp_string + self.output_format
    else:
      self.log_file = log_file

  def main(self):
    first = True
    print "Saver: PID", os.getpid()
    while True:
      try:
        Data = self.inputs[0].recv()  # recv data
        data = np.transpose(Data.values())
        fo = open(self.log_file, "a", buffering=3)  # "a" for appending, buffering to limit r/w disk access.
        fo.seek(0, 2)  # place the "cursor" at the end of the file

        if first and not self.existing:
          legend_ = Data.keys()
          if self.output_format == '.txt':
            fo.write(str([legend_[i] for i in range(len(legend_))]) + "\n")
          elif self.output_format == '.csv':
            #  Write the first line : value, value, value,
            for index in xrange(len(legend_) - 1):
              fo.write(str(legend_[index]) + ',')
            # Write the last line : value \n
            fo.write(str(legend_[-1]) + '\n')
          first = False

        if self.output_format == '.txt':
          data_to_save = str(data) + "\n"
          fo.write(data_to_save)
          fo.close()

        elif self.output_format == '.csv':
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
