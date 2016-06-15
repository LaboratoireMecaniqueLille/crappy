from crappy.technical import __boardnames__ as boardnames


class Acquisition:
    def __init__(self, port='/dev/ttyS0', baudrate=9600, board_name="comedi"):

        self.baudrate = baudrate
        self.port = port
        if not board_name.capitalize() in [x.capitalize() for x in boardnames]:
            while not board_name.capitalize() in [x.capitalize() for x in boardnames]:
                print "Unreconized io board: %s" % board_name
                print "board_name should be one of these: "
                print boardnames, '\n'
                board_name = raw_input('board_name (ENTER to resume): ')
                print '\n'
                if len(board_name) == 0:
                    print "Unreconized io board name, leaving program..."
                    return
        try:
            module = __import__("crappy.sensor", fromlist=["%sSensor" % boardnames[0]])
            board_name = boardnames[[x.capitalize() for x in boardnames].index(board_name.capitalize())]
            self.sensor = getattr(module, "%sSensor" % board_name)
        except Exception as e:
            print "{0}".format(e), " : Unreconized io board name, leaving program...\n"
            return

    def close(self):
        self.sensor.close()

    def new(self):
        self.sensor.new()
        pass

    def getData(self):
        return self.sensor.get_data()
