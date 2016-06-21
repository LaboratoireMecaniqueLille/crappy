# coding: utf-8
import time

from _meta import MasterBlock


class CommandTriboMaintien(MasterBlock):

    def __init__(self, VariateurTribo, comediDigital, comediOut):
        super(CommandTriboMaintien, self).__init__()
        self.VariateurTribo = VariateurTribo

    def main(self):
        t_0 = time.time()
        print "t0 = ", t_0
        datastring = ''
        self.VariateurTribo.init = False
        consigne = 0.0
