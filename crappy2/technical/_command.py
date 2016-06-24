from crappy2.technical import __boardnames__ as board_names


class Command:
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
            module = __import__("crappy2.actuator", fromlist=["%sActuator" % board_names[0]])
            board_name = board_names[[x.capitalize() for x in board_names].index(board_name.capitalize())]
            actuator = getattr(module, "%sActuator" % board_name)
            self.actuator = actuator(*args, **kwargs)
        except Exception as e:
            print "{0}".format(e), " : Unreconized io board name, leaving program...\n"
            return

    def close(self):
        self.actuator.close()

    def new(self):
        self.actuator.new()
        pass

    def set_cmd(self, cmd):
        return self.actuator.set_cmd(cmd)
