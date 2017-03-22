# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup PID PID
# @{

## @file pidtomo.py
# @brief input target value for effort at some speed. Actuator will go to effort traget value at that speed. Not much of a pid ain't it ?
# @author Grégory Hauss
# @version 0.1
# @date 13/02/2017


from .masterblock import MasterBlock
from ..tool import DataPicker
import os

class PIDTomo(MasterBlock):
    def __init__(self,technical):
        """
        Read the target value and goes for it at defined speed once you click on PIDStart. Amazing.
        """
        super(PIDTomo, self).__init__()
        print "pidtomo !", os.getpid()
        self.technical = technical


    def main(self):
        datapicker = DataPicker(self.inputs[1])
        Data = datapicker.get_data()
        #nothingtodohere=self.inputs[0].recv()
        #print "data", Data
        #print "input0",nothingtodohere
        try:
            while True:
                #print "hello world"
                PIDOptions = self.inputs[0].recv()
                ########################################DENIED, quand saver est actif, la commade self.inputs[0].recv() ne fonctionne pas.
                #print "wassup ?"
                #print PIDOptions['PIDFlag']
                if PIDOptions['PIDFlag'] == 1:
                    speedpid= PIDOptions['SpeedPID']
                    loadpid = PIDOptions['LoadPID']
                    #print speedpid, loadpid
                    Data = datapicker.get_data()
                    data = Data.values()
                    load = float(data[2])
                    #print load,loadpid,speedpid
                    #traction
                    if load>loadpid:
                        self.technical.set_speed(-speedpid)
                    #compression
                    elif load<loadpid:
                        self.technical.set_speed(speedpid)
                    #consigne atteinte
                    else: self.technical.set_speed(0)
            datapicker.close()
        except KeyboardInterrupt:
            print "KeyboardInterrupt received in PIDTomo"
            datapicker.close()
             #raise
        except Exception as e:
            print "Exception in PIDTomo %s: %s" % (os.getpid(), e)
            datapicker.close()
            # raise
            # time.sleep(1)
