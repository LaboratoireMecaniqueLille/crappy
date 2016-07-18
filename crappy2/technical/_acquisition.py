# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup Acquisition Acquisition
# @{

## @file _acquisition.py
# @brief  Acquisition class to use all implemented sensors.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

from crappy2.technical import __boardnames__ as board_names


class Acquisition:
    def __init__(self, board_name="comedi", *args, **kwargs):

        if not board_name.capitalize() in [x.capitalize() for x in board_names]:
            while not board_name.capitalize() in [x.capitalize() for x in board_names]:
                print "Unreconized io board: %s" % board_name
                print "board_name should be one of these: "
                print board_names, '\n'
                board_name = raw_input('board_name (ENTER to resume): ')
                print '\n'
                if len(board_name) == 0:
                    print "Unreconized io board name, leaving program..."
                    return
        try:
            module = __import__("crappy2.sensor", fromlist=["%sSensor" % board_names[0]])
            board_name = board_names[[x.capitalize() for x in board_names].index(board_name.capitalize())]
            sensor = getattr(module, "%sSensor" % board_name)
            self.sensor = sensor(*args, **kwargs)
        except Exception as e:
            print "{0}".format(e), " : Unreconized io board name, leaving program...\n"
            return

    def close(self):
        self.sensor.close()

    def new(self):
        self.sensor.new()
        pass

    def get_data(self, *args, **kwargs):
        return self.sensor.get_data(*args, **kwargs)
