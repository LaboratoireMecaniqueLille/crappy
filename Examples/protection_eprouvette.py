#coding: utf-8

import crappy

if __name__ == "__main__":
  
      instronSensor = crappy.sensor.ComediSensor(channels=[1, 3], gain=[-3749.3, -3198.9 * 1.18], offset=[0, 0])
      t, F2 = instronSensor.get_data(0)
      t, F4 = instronSensor.get_data(1)
      print "Offset",F2,F4
      instronSensor = crappy.sensor.ComediSensor(channels=[1, 3], gain=[-3749.3, -3198.9 * 1.18], offset=[-F2, -F4])
      biaxeTech1 = crappy.technical.Biaxe(port='/dev/ttyS4')
      biaxeTech2 = crappy.technical.Biaxe(port='/dev/ttyS5')
      biaxeTech3 = crappy.technical.Biaxe(port='/dev/ttyS6')
      biaxeTech4 = crappy.technical.Biaxe(port='/dev/ttyS7')
      axes = [biaxeTech1, biaxeTech2, biaxeTech3, biaxeTech4]
      graph_effort = crappy.blocks.Grapher(('t(s)', 'F2(N)'), ('t(s)', 'F4(N)'))

      effort = crappy.blocks.MeasureByStep(instronSensor, labels=['t(s)', 'F2(N)', 'F4(N)'], freq=100,compacter=20)


      signalGenerator = crappy.blocks.SignalGenerator(
          path=[{"waveform": "protection", "gain": 1, "lower_limit": [-1, 'F2(N)'], "upper_limit": [10, 'F2(N)']}],
          send_freq=100, repeat=True)

      signalGenerator_horizontal = crappy.blocks.SignalGenerator(
          path=[{"waveform": "protection", "gain": 1, "lower_limit": [-1, 'F4(N)'], "upper_limit": [10, 'F4(N)']}],
          send_freq=100, repeat=True)

      biotens = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech1, biaxeTech2], speed=-5000)  # vertical
      biotens_horizontal = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech3, biaxeTech4], speed=-5000)

      crappy.link(effort,signalGenerator)
      crappy.link(effort,signalGenerator_horizontal)
      crappy.link(signalGenerator,biotens)
      crappy.link(signalGenerator_horizontal,biotens_horizontal)
      crappy.link(effort,graph_effort)




      crappy.start()

