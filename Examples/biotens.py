#coding: utf-8
from __future__ import print_function

import time
import numpy as np
import crappy

def eval_offset(device, duration):
    """
    Method to evaluate offset. Returns an average of measurements made during specified time.
    """
    timeout = time.time() + duration  # secs from now
    print('Measuring offset (%d sec), please wait...'%duration)
    offset = []
    while True:
      chan1 = device.get_data('all')[1]
      offset.append(chan1)
      if time.time() > timeout:
        offsets = -np.mean(offset)
        print('offset:', offsets)
        break
    return offsets

if __name__ == "__main__":
    save_path = "biotens_data/"
    timestamp = time.ctime()[:-5].replace(" ","_")
    save_path += timestamp+"/"
    # Creating F sensor
    F_sensor = crappy.sensor.ComediSensor(channels=[0], gain=[-48.8], offset=[0])
    F_offset = eval_offset(F_sensor,1)
    F_sensor = crappy.sensor.ComediSensor(channels=[0], gain=[-48.8], offset=[F_offset])
    # Creating F block
    effort = crappy.blocks.MeasureByStep(F_sensor, labels=['t(s)', 'F(N)'], freq=100)
    # Associated compacter
    comp_effort = crappy.blocks.Compacter(100)
    crappy.link(effort,comp_effort)
    # grapher
    graph_effort = crappy.blocks.Grapher(('t(s)','F(N)'),length=30)
    crappy.link(comp_effort,graph_effort)
    # and saver
    save_effort = crappy.blocks.Saver(save_path+"effort.csv")
    crappy.link(comp_effort,save_effort)
    
    # Creating biotens technical
    biotensTech = crappy.technical.Biotens(port='/dev/ttyUSB0', size=10)  # Used to initialize motor.
    # Biotens block
    biotens = crappy.blocks.CommandBiotens(biotens_technicals=[biotensTech], speed=5)
    # Position compacter
    comp_pos= crappy.blocks.Compacter(5)
    crappy.link(biotens,comp_pos)
    # grapher
    graph_pos= crappy.blocks.Grapher(('t(s)', 'position'), length=10)
    crappy.link(comp_pos,graph_pos)
    # And saver
    save_pos= crappy.blocks.Saver(save_path+'pos.csv')
    crappy.link(comp_pos,save_pos)

    # To pilot the biotens
    signal_generator = crappy.blocks.SignalGenerator(path=[
      {"waveform": "limit", "gain": 1, "cycles": 2, "phase": 0, "lower_limit": [0.02, 'F(N)'],
       "upper_limit": [30, 'F(N)']}],
      send_freq=5, repeat=False, labels=['t(s)', 'signal', 'cycle']) 
    crappy.link(effort,signal_generator)
    crappy.link(signal_generator,biotens)



    # VideoExtenso
    extenso = crappy.blocks.VideoExtenso(camera="Ximea", white_spot=False, display=True)
    # Compacter
    comp_extenso = crappy.blocks.Compacter(90)
    crappy.link(extenso,comp_extenso)
    # Saver
    save_extenso = crappy.blocks.Saver(save_path+'extenso.csv')
    crappy.link(comp_extenso, save_extenso)
    # And grapher
    graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
    crappy.link(comp_extenso, graph_extenso)

    #And here we go !
    crappy.start()
