# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup BiaxeActuator BiaxeActuator
# @{

## @file _biaxeActuator.py
# @brief  Declare a new axis for the Biaxe.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

import serial
from ._meta import motion
from .._warnings import deprecated as deprecated


# Parameters
# limit = 0.0005 # limit for the eprouvette protection
# offset_=-0.0056
# protection_speed=1000. # nominal speed for the protection
# frequency=500. # refreshing frequency (Hz)
# alpha = 1.05

class BiaxeActuator(motion.MotionActuator):
    """Declare a new axis for the Biaxe"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=38400, timeout=1, ser=None, **kwargs):
        ## @fn __init__()
        # @brief This class create an axis and opens the corresponding serial port.
        #
        # @param port : str
        #         Path to the corresponding serial port, e.g '/dev/ttyS4'
        # @param baudrate : int, default = 38400
        #         Set the corresponding baud rate.
        # @param timeout : int or float, default = 1
        #         Serial timeout.
        super(BiaxeActuator, self).__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        if 'baud_rate' in kwargs:
            print 'WARNING: "baud_rate" keyword is deprecated, use "baudrate" instead'
            self.baudrate = baudrate

        if ser != None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, self.baudrate,
                                     serial.EIGHTBITS, serial.PARITY_EVEN
                                     , serial.STOPBITS_ONE, self.timeout)
            self.ser.write("OPMODE 0\r\n EN\r\n")

    def set_speed(self, speed):
        """Re-define the speed of the motor. 1 = 0.002 mm/s"""
        # here we should add the physical conversion for the speed
        self.ser.write("J " + str(speed) + "\r\n")

    def set_position(self, position, speed, motion_type='relative'):
        # TODO
        pass

    """Reset the position to zero"""

    def move_home(self):
        # TODO
        pass

    @deprecated(None, "serial port is now initialized in __init__")
    def new(self):
        ## @fn new()
        # @brief DEPRECATED: serial port is now initialized in __init__
        # No arguments, open port, set speed mode and engage
        #
        pass

    @deprecated(None, "replaced by close method in _biaxeTechnical")
    def close_port(self):
        #
        # DEPRECATED: replaced by close method in _biaxeTechnical.
        # Close the designated port
        #
        self.ser.close()

    @deprecated(None)
    def CLRFAULT(self):
        """Reset errors"""
        self.ser.write("CLRFAULT\r\n")
        self.ser.write("OPMODE 0\r\n EN\r\n")

    # def protection_eprouvette(Vmax,*args):
    # This function aim to keep the sensor value at the same level as the initial level,
    # and moves the motor in consequence.
    # args must be open Ports, paired with the corresponding sensor,
    # and data pipes e.g. for each port: [port0, axe0,time_pipe,sensor_pipe,speed_pipe]"""
    # condition=True
    # speed=0
    # speed_i=np.zeros(len(args))
    # offset=np.zeros(len(args))
    # for i in range(len(args)):
    # print "Evaluating offset for port %s..." %i
    # for j in range(int(1*frequency)):
    # t_sensor, effort=args[i][1].get()
    # offset[i]+=effort/(1.*frequency)
    # print "Done : offset for port %s = %s" %(i,offset[i])
    ##time.sleep(10)
    # t0=time.time()  #define origin of time for this test
    # t=t0
    # while condition==True:
    # while (time.time()-t)<(1./(frequency*len(args))):
    # indent=True
    # t=time.time()
    # for i in range(len(args)):
    # t_sensor, effort=args[i][1].get()
    ##print "i= %s, effort = %s" %(i,effort)
    # t_sensor-=t0 # use t0 as origin of time
    # if (effort-offset[i]) >= limit:
    # speed=-Vmax
    # elif (effort-offset[i]) <= -limit:
    # speed=Vmax
    # else:
    # speed=0
    # if speed!=speed_i[i]:
    # args[i][0].move(speed)
    ##print "speed = %s" %speed
    # speed_i[i]=speed
    # args[i][2].send(t_sensor) # send data to the save function
    # args[i][3].send(effort-offset[i])
    # args[i][4].send(speed)


    # def etalonnage(time_pipe,jauge_pipe,F0_pipe,F1_pipe,ports,axes,jauge,Fmax,Fmin,Vmax):
    # speed_i=0
    # t0_,V=jauge()
    # print "jauge = %s" %V
    # print "Fmax=%s" %Fmax
    # print "Fmin = %s" %Fmin
    # offset_F0=0
    # offset_F1=0
    # for i in range(100): # 100 points - mean of the minimal Tension
    # t,F0=axes[0]()
    # offset_F0+=F0/100.
    # for i in range(100): # 100 points - mean of the minimal Tension
    # t,F1=axes[1]()
    # offset_F1+=F1/100.
    # while V >= Fmax:
    # print "1"
    # speed=-Vmax
    # t,V=jauge()
    # print V
    # t,F1=axes[1]()
    # t,F0=axes[0]()
    # time_pipe.send(t-t0_)
    # jauge_pipe.send(V)
    # F0_pipe.send(F0-offset_F0)
    # F1_pipe.send(F1-offset_F1)
    # if speed!=speed_i:
    # speed_i=speed
    # ports[0].move(speed)
    # ports[1].move(speed)
    # while V <= Fmin:
    # print "2"
    # speed=Vmax
    # t,V=jauge()
    # print V
    # t,F1=axes[1]()
    # t,F0=axes[0]()
    # time_pipe.send(t-t0_)
    # jauge_pipe.send(V)
    # F0_pipe.send(F0-offset_F0)
    # F1_pipe.send(F1-offset_F1)
    # if speed!=speed_i:
    # speed_i=speed
    # ports[0].move(speed)
    # ports[1].move(speed)
    # ports[0].move(0)
    # ports[1].move(0)
