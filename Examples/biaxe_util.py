#coding: utf-8
from __future__ import print_function

import crappy2 as crappy

biaxeTech = []
biaxeTech.append(crappy.technical.Oriental(port='/dev/ttyUSB0'))
biaxeTech.append(crappy.technical.Oriental(port='/dev/ttyUSB1'))
biaxeTech.append(crappy.technical.Oriental(port='/dev/ttyUSB2'))
biaxeTech.append(crappy.technical.Oriental(port='/dev/ttyUSB3'))

motors = ['A','B','C','D']
biaxeTech = sorted(biaxeTech,key=lambda x:x.num_device)
biaxeDict = {}
i=0
for c in 'abcd':
  biaxeDict[c] = [biaxeTech[i]]
  i+=1
biaxeDict['x'] = biaxeDict['b']+biaxeDict['d']
biaxeDict['y'] = biaxeDict['a']+biaxeDict['c']

for i in range(4):
  print("BiaxeTech{0} (on ttyUSB{0}) is motor {1}".format(
            i,motors[biaxeTech[i].num_device-1]))

user_input = ''
while True:
  print("Axe and speed ?>",end="")
  user_input = raw_input().lower()
  if user_input == 'q':
    break
  try:
    axe = user_input[:1]
    speed = int(user_input[1:].strip())
  except ValueError:
    if user_input == 'clear':
      for axe in biaxeTech:
        axe.clear_errors()
      continue
    print("Unknow command, stopping")
    for axe in biaxeTech:
      axe.actuator.set_speed(0)
    continue

  if user_input[0] in 'xyabcd':
    for axe in biaxeDict[user_input[0]]:
      axe.actuator.set_speed(speed)
