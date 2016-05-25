# coding: utf-8
#import serial
import time
from ._meta import motion
from .._deprecated  import _deprecated as deprecated

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

class Lal300Sensor(motion.MotionSensor):
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

    def get_position(self):#Releve de la position du moteur via le port serie
        """Check current position."""
        self.ser.read(self.ser.in_waiting) #Nettoyage du port serie
        time.sleep(0.015)
        self.ser.write('TP\r\n') #Ecriture de l'instruction "Tell Position" pour indiquer la position moteur
        time.sleep(0.013) #Temporisation pour permettre l'ecriture de la consigne sur le port serie
        self.ser.read(4) #Lire et effacer les 4 caracteres donnes par l'instruction precedente
        time.sleep(0.013)
        d=self.ser.read(self.ser.in_waiting-3) #Lire et effacer les tous caracteres donnes par l'instruction precedente sauf les 3 derniers correspondant au \r \n et au retour a la ligne
        time.sleep(0.015)
        disp=int(d) #Conversion en integer de la position lue via le port serie
        return disp

    @deprecated(get_position)
    def checkdisp(self):
        #DEPRECATED: use get_position instead.
        #Releve de la position du moteur via le port serie
        """Check current position."""
        return self.get_position()

class SensorLal300(Lal300Sensor):
    @deprecated(None, "Use Lal300Sensor class instead.")
    def __init__(self,param,ser):
        #DEPRECATED: Use Lal300Sensor class instead.
        # Arguments param et ser indiques dans la classe TechnicalLal300
        """This class contains methods to get info from the motors of the lal300
        machine. You should NOT use it directly, but use the Lal300Technical.
        """
        super(SensorLal300, self).__init__(ser=ser, param=param)


