from crappy.technical import __boardnames__ as boardnames

class Command:

    def __init__(self, port='/dev/ttyS0', baudrate=9600, board_name="comedi"):
        
        if not board_name.capitalize() in [x.capitalize() for x in boardnames]:
            while not board_name.capitalize() in [x.capitalize() for x in boardnames]:
                print "Unreconized io board: %s"%board_name
                print "board_name should be one of these: "
                print boardnames, '\n'
                board_name = raw_input('board_name (ENTER to resume): ')
                print '\n'
                if len(board_name) == 0:
                     print "Unreconized io board name, leaving program..."
                     return
        try:
            module = __import__("crappy.actuator", fromlist=["%sActuator"%boardnames[0]])
            board_name = boardnames[[x.capitalize() for x in boardnames].index(board_name.capitalize())]
            self.actuator= getattr(module, "%sActuator"%board_name)
        except Exception as e:
            print "{0}".format(e), " : Unreconized io board name, leaving program...\n"
            return
        pass

    def close(self):
        self.actuator.close()
    
    def new(self):
        self.actuator.new()
        pass
    
    def set_cmd(self):
        return self.actuator.set_cmd()
    