# coding: utf-8
import crappy2
from time import sleep

labjack_debit = crappy2.actuator.LabJackActuator(channel='FIO3')  # Connected to YO1
labjack_verin = crappy2.actuator.LabJackActuator(channel='FIO2')  # Connected to YO2

def cmd_hydrau(state):
    
    tempo_verin = 0.2
    tempo_debit_on = 0.2
    if state:

        labjack_verin.set_cmd(1)
        labjack_debit.set_cmd(1)  # Debit ON
        sleep(tempo_verin)        # temps de sortie du vérin (a tester!!!)
        labjack_debit.set_cmd(0)  # Debit OFF 

    if not state:
        labjack_debit.set_cmd(1)  # Debit ON
        sleep(tempo_debit_on)  # Temps d'établissement du débit dans le circuit (a tester)
        labjack_verin.set_cmd(0)
        sleep(tempo_verin)  
        labjack_debit.set_cmd(0)  # Debit OFF

def close():
    labjack_debit.set_cmd(0)
    labjack_verin.set_cmd(0)
    labjack_debit.close()
