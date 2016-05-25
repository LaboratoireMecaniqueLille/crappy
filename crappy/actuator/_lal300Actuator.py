# coding: utf-8
import serial
import time
from .._deprecated  import _deprecated as deprecated
from ._meta import motion

p = {}
p['timeout'] = 0.#s
n = 3 # modify with great caution
p['PID_PROP'] = 8/n
p['PID_INT'] = 30/n
p['PID_DERIV'] = 200/n
p['PID_INTLIM'] = 1000/n
p['ACC'] = 6000.
p['ACconv'] = 26.22#conversion ACC values to mm/s/s
p['FORCE'] =30000.
p['SPEEDconv'] = 131072.#conversion SPEED values to mm/s
p['ENTREE_VERIN']='DI1'
p['SORTIE_VERIN']='DI0'

class Lal300Actuator(motion.MotionActuator):
    def __init__(self, param=p, ser=None, port='/dev/ttyUSB1', baudrate=19200):
        """
        Open the connection, and initialise the Lal300.

        Parameters
        ----------
        param : dict
            Dict of parameters.

                * 'port' : str
                        Path to the serial port.
                * 'baudrate' : int
                        Corresponding baudrate.
                * 'timeout' : float
                        Timeout of the serial connection.
                * 'PID_PROP' : float
                        Proportionnal coefficient of the PID.
                * 'PID_INT' : float
                        Integral coefficient for the PID.
                * 'PID_DERIV' : float
                        Derivative coefficient for the PID.
                * 'PID_INTLIM' : float
                        Limit of the integral coefficient.
                * 'ACC' float
                        Acceleration of the motor.
                * 'ACconv' : float 
                        Conversion ACC values to mm/s/s
                * 'FORCE' : float 
                        Maximal force provided by the motor.
                * 'SPEEDconv' : float 
                        Conversion SPEED values to mm/s
                * 'ENTREE_VERIN' : str
                        'DI1'
                * 'SORTIE_VERIN' : str
                        'DI0'
                * 'ETIRE': list of int
                        List of extreme values for the position in traction.
                * 'COMPRIME': list of int
                        List of extreme values for the position in compression.
                * 'SPEED' : list of int
                        List of speed, for each group of cycles.
                * 'CYCLES' : list of int
                        List of cycles, for each group.
        """
        self.param=param
        self.port = port
        self.baudrate = baudrate
        if ser != None:
            self.ser = ser
        else:
            self.ser=serial.Serial(port=self.port, #Configuration du port serie à l'aide de PySerial
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=param['timeout'],
            rtscts=False,
            write_timeout=None,
            dsrdtr=False,
            inter_byte_timeout=None)

    
    def set_speed(self, speed):
        #TODO
        pass
    
    def set_position(self,consigne): #Envoi de la consigne de position au moteur
        """Send a position command to the motor"""
        self.ser.write(consigne) #Ecriture de la consigne dans le port serie	
        time.sleep(0.005)
        self.ser.read(self.ser.in_waiting)
        time.sleep(0.025) #Temporisation pour permettre l'effacement complet des commandes stockees dans le port serie
        
    def move_home(self): #Deplacement au point d'origine "Home"
        """Go back to the homing point"""
        self.ser.write("PM,MA-10,GO\r\n")
        time.sleep(0.005)
        self.ser.read(self.ser.in_waiting)
        return self.ser

    @deprecated(None, 'Use reset method defined in Lal300Technical instead')
    def reset(self): 
        #Reinitialisation des parametres du correcteur PID du moteur et defintion de l'origine moteur
        """
        DEPRECATED: Use reset method defined in Lal300Technical instead.
        Reset PID parameters and re-defined the motor origin
        """
        self.ser.write("MF,RM,SG%i,SI%i,SD%i,IL%i,DH\r\n"%(self.param['PID_PROP'],self.param['PID_INT'],self.param['PID_DERIV'],self.param['PID_INTLIM'])) #set PID values valeur DH modifiee
        time.sleep(0.005) #Temporisation assurant la bonne ecriture de la ligne precedente dans le port serie
        self.ser.read(self.ser.in_waiting) # Nettoyage du port serie
        time.sleep(0.005)
        
    @deprecated(move_home)
    def homing(self): 
        #Deplacement au point d'origine "Home"
        """
        DEPRECATED: use move_home instead.
        Go back to the homing point
        """
        self.ser.write("PM,MA-10,GO\r\n")
        time.sleep(0.005)
        self.ser.read(self.ser.in_waiting)
        return self.ser

    @deprecated(None, 'Use stop method defined in Lal300Technical instead')
    def stoplal300(self): 
        #Arret du moteur
        """
        DEPRECATED: Use stop method defined in Lal300Technical instead.
        Stop the motor
        """
        self.ser.write("MF\r\n") #Envoi de la commande "Moteur OFF" via le port serie
        time.sleep(0.005)
        self.ser.read(self.ser.in_waiting) 

    @deprecated(None, 'Use close method defined in Lal300Technical instead')
    def closelal300(self):
        #Fermeture du port serie
        """
        DEPRECATED: Use close method defined in Lal300Technical instead.
        Close serial port
        """
        self.stoplal300()#Arret du moteur
        self.ser.close()  #Fermeture du port serie
        return self.ser.isOpen() #Verification de l'ouverture du port serie (True/False)

class ActuatorLal300(Lal300Actuator):
    @deprecated(None, 'use Lal300Actuator class instead.')
    def __init__(self,param,ser): 
        # Arguments param et ser indiques dans la classe TechnicalLal300
        """
        DEPRECATED: use Lal300Actuator class instead.
        This class contains methods to command the motor of the Lal300.
        You should NOT use it directly, but use the Lal300Technical.
        """
        #print param, ser
        #self.param=param
        #self.ser=ser
        super(ActuatorLal300, self).__init__(ser=ser, param=param)