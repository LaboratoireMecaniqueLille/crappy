# coding: utf-8
# import time
import serial

from ._meta import motion
from .._warnings import deprecated
from ..actuator import Lal300Actuator
from ..sensor import Lal300Sensor
import time

n = 3  # modify with great caution
p = {'timeout': 0., 'PID_PROP': 8 / n, 'PID_INT': 30 / n, 'PID_DERIV': 200 / n, 'PID_INTLIM': 1000 / n, 'ACC': 6000.,
     'ACconv': 26.22, 'FORCE': 30000., 'SPEEDconv': 131072., 'ENTREE_VERIN': 'DI1', 'SORTIE_VERIN': 'DI0'}


class Lal300(motion.Motion):
    def __init__(self, param=p, port='/dev/ttyUSB1', baudrate=19200):
        """
        Open the connection, and initialise the Lal300.

        You should always use this Class to communicate with the Lal300.

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
        Examples
        --------
        param = {}
        param['timeout'] = 0.#s
        n = 3 # modify with great caution
        param['PID_PROP'] = 8/n
        param['PID_INT'] = 30/n
        param['PID_DERIV'] = 200/n
        param['PID_INTLIM'] = 1000/n
        param['ACC'] = 6000.
        param['ACconv'] = 26.22#conversion ACC values to mm/s/s
        param['FORCE'] =30000.
        param['SPEEDconv'] = 131072.#conversion SPEED values to mm/s
        param['ENTREE_VERIN']='DI1'
        param['SORTIE_VERIN']='DI0'
        ##### modifiable values :
        param['ETIRE']=[-900,-1000,-1100,-1200,-2400,-3600,-4800,-6000,-7200,-8400,-9800,-11000,-12000,-18000,-24000,-36000,-48000,-60000,-72000,-84000]
        param['COMPRIME']=[-200,-300,-400,-500,-700,-800,-900,-900,-1500,-3000,-3000,-5000,-5000,-5000,-5000,-5000,-5000,-5000,-5000,-5000]
        param['SPEED'] = [15000,15000,15000,16000,30000,45000,80000,110000,130000,150000,180000,210000,250000,300000,350000,400000,500000,550000,600000,650000]
        param['CYCLES']=[2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000]
        """
        super(Lal300, self).__init__(port, baudrate)
        self.param = param
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port=self.port,  # Configuration du port serie à l'aide de PySerial
                                 baudrate=self.baudrate,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=param['timeout'],
                                 rtscts=False,
                                 write_timeout=None,
                                 dsrdtr=False,
                                 inter_byte_timeout=None)
        self.actuator = Lal300Actuator(self.param,
                                       self.ser)  # Appel de la sous-classe ActuatorLal300 avec les parametres situes dans le programme lal300Main.py
        self.sensor = Lal300Sensor(self.param,
                                   self.ser)  # Appel de la sous-classe SensorLal300 avec les parametres situes dans le programme lal300Main.py

    def stop(self):  # Arret du moteur
        """Stop the motor"""
        self.ser.write("MF\r\n")  # Envoi de la commande "Moteur OFF" via le port serie
        time.sleep(0.005)
        self.ser.read(self.ser.in_waiting)

    def close(self):  # Fermeture du port serie
        """Close serial port"""
        self.stop()  # Arret du moteur
        self.ser.close()  # Fermeture du port serie
        return self.ser.isOpen()  # Verification de l'ouverture du port serie (True/False)

    def reset(self):  # Reinitialisation des parametres du correcteur PID du moteur et defintion de l'origine moteur"""
        """Reset PID parameters and re-defined the motor origin"""
        self.ser.write("MF,RM,SG%i,SI%i,SD%i,IL%i,DH\r\n" % (
            self.param['PID_PROP'], self.param['PID_INT'], self.param['PID_DERIV'],
            self.param['PID_INTLIM']))  # set PID values valeur DH modifiee
        time.sleep(0.005)  # Temporisation assurant la bonne ecriture de la ligne precedente dans le port serie
        self.ser.read(self.ser.in_waiting)  # Nettoyage du port serie
        time.sleep(0.005)

    def clear_errors(self):
        # TODO
        pass


class TechnicalLal300(Lal300):
    """
    DEPRECATED: use Lal300 class instead.
    Open both a Lal300Sensor and Lal300Actuator instances.
    """

    @deprecated(None, "Use Lal300 class instead.")
    def __init__(self, param):
        """
        Open the connection, and initialise the Lal300.
        """
        super(TechnicalLal300, self).__init__(param, param['port'], param['baudrate'])
