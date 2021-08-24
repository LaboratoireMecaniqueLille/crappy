# coding: utf-8

import crappy

help_string = """
Syntax:
{axe}{speed}
(You can put a space between axe and speed)
Example:
  x -5
Will pull the axe in at 5mm/s
  a20
Will push out motor A only (the number 1)
note: on this machine, motors are numbered from 1 to 4. Here they are labelled
a,b,c,d. (a is 1, b is 2,...)

Use q to quit
Type 'h' or 'help' to see this message again
"""


actuator_list = list()
actuator_list.append(crappy.actuator.Biaxe(port='/dev/ttyS4'))  # Motor 1
actuator_list.append(crappy.actuator.Biaxe(port='/dev/ttyS5'))  # Motor 2
actuator_list.append(crappy.actuator.Biaxe(port='/dev/ttyS6'))  # Motor 3
actuator_list.append(crappy.actuator.Biaxe(port='/dev/ttyS7'))  # Motor 4
for act in actuator_list:
  act.open()

actuator_dict = {}
i = 0
for c in 'abcd':
  actuator_dict[c] = [actuator_list[i]]
  i += 1
actuator_dict['x'] = actuator_dict['c']+actuator_dict['d']
actuator_dict['y'] = actuator_dict['a']+actuator_dict['b']

print(help_string)

user_input = ''
while True:
  user_input = input("Axe and speed ?> ").lower()
  if user_input == 'q':
    break
  elif user_input in ['h', 'help', '?']:
    print(help_string)
    continue
  try:
    axe = user_input[:1]
    speed = float(user_input[1:].strip())
  except ValueError:
    print("Unknown command, stopping")
    for axe in actuator_list:
      axe.set_speed(0)
    continue

  if user_input[0] in 'xyabcd':
    for axe in actuator_dict[user_input[0]]:
      axe.set_speed(speed)
