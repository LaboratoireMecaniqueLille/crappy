#coding: utf-8
from __future__ import print_function

import crappy


help_string = """
Syntax:
{axe}{speed}
(You can put a space between axe and speed)
Example:
  x -50
Will pull the axe in at 50% of the speed
  a20
Will push out motor A only (the one with numdevice=1)

Use q to quit
Use clear to clear the errors on all the motors
Type 'h' or 'help' to see this message again
"""



biaxeTech = []
biaxeTech.append(crappy.actuator.Oriental(port='/dev/ttyUSB0'))
biaxeTech.append(crappy.actuator.Oriental(port='/dev/ttyUSB1'))
biaxeTech.append(crappy.actuator.Oriental(port='/dev/ttyUSB2'))
biaxeTech.append(crappy.actuator.Oriental(port='/dev/ttyUSB3'))

motors = ['A','B','C','D']
biaxeTech = sorted(biaxeTech,key=lambda x:x.num_device)
biaxeDict = {}
i=0
for c in 'abcd':
  biaxeDict[c] = [biaxeTech[i]]
  i+=1
biaxeDict['x'] = biaxeDict['b']+biaxeDict['d']
biaxeDict['y'] = biaxeDict['a']+biaxeDict['c']

print(help_string)
for i in range(4):
  print("BiaxeTech{} (on {}) is motor {}".format(
            i,biaxeTech[i].port,motors[biaxeTech[i].num_device-1]))

user_input = ''
while True:
  print("Axe and speed ?>",end="")
  user_input = raw_input().lower()
  if user_input == 'q':
    break
  elif user_input in ['h','help','?']:
    print(help_string)
    continue
  elif user_input == 'clear':
    for axe in biaxeTech:
      axe.clear_errors()
    continue
  try:
    axe = user_input[:1]
    speed = int(user_input[1:].strip())
  except ValueError:
    print("Unknow command, stopping")
    for axe in biaxeTech:
      axe.set_speed(0)
    continue

  if user_input[0] in 'xyabcd':
    for axe in biaxeDict[user_input[0]]:
      axe.set_speed(speed)
