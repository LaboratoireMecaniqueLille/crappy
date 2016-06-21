# coding: utf-8
from _meta import MasterBlock


class CommandPI(MasterBlock):
    """Receive a signal and send it for the PI actuator"""

    def __init__(self, PI_actuators, signal_label='signal'):
        """
        Receive a signal and translate it for the PI actuator.

        Parameters
        ----------
        PI_actuators : crappy2.actuators.PIactuators objects
            Axe to control.
        signal_label : str, default = 'signal'
            Label of the data to be transfered.
        """
        super(CommandPI, self).__init__()
        self.PI_actuators = PI_actuators
        self.signal_label = signal_label

    def main(self):
        try:
            last_cmd = -118163210
            while True:
                Data = self.inputs[0].recv()
                # try:
                # cmd=Data['signal'].values[0]
                # except AttributeError:
                cmd = Data[self.signal_label]
                if cmd != last_cmd:
                    self.PI_actuators.set_absolute_disp(cmd)
                    last_cmd = cmd
                # for debugging :
                # self.PI_actuators.ser.write("%c%cTP\r"%(1,'0')) #connaitre la position
                # print "position platine : ", self.PI_actuators.ser.readline()
                # print "commande platine : ", cmd
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in CommandPI : ", e
            self.PI_actuators.close_port()
        # raise
