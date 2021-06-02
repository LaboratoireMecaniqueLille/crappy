# coding: utf-8

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


actuator_list = list()
actuator_list.append(crappy.actuator.Oriental(port='/dev/ttyUSB0'))
actuator_list.append(crappy.actuator.Oriental(port='/dev/ttyUSB1'))
actuator_list.append(crappy.actuator.Oriental(port='/dev/ttyUSB2'))
actuator_list.append(crappy.actuator.Oriental(port='/dev/ttyUSB3'))
for act in actuator_list:
  act.open()

motors = ['A', 'B', 'C', 'D']
actuator_list = sorted(actuator_list, key=lambda x: x.num_device)
actuator_dict = {}
i = 0
for c in 'abcd':
  actuator_dict[c] = [actuator_list[i]]
  i += 1
actuator_dict['x'] = actuator_dict['b']+actuator_dict['d']
actuator_dict['y'] = actuator_dict['a']+actuator_dict['c']

actuator_dict['z'] = actuator_dict['x']+actuator_dict['y']

print(help_string)
for i in range(4):
  print("BiaxeTech{} (on {}) is motor {}".format(
            i, actuator_list[i].port, motors[actuator_list[i].num_device - 1]))

user_input = ''
while True:
  print("Axe and speed ?>", end="")
  try:
    user_input = input().lower()
  except (EOFError, KeyboardInterrupt):
    break
  if user_input == 'q':
    break
  elif user_input in ['h', 'help', '?']:
    print(help_string)
    continue
  elif user_input == 'clear':
    for axe in actuator_list:
      axe.clear_errors()
    continue
  try:
    axe = user_input[:1]
    speed = int(user_input[1:].strip()) * .07  # From % to mm/min
  except ValueError:
    print("Unknown command, stopping")
    for axe in actuator_list:
      axe.set_speed(0)
    continue

  if user_input[0] in 'xyzabcd':
    for axe in actuator_dict[user_input[0]]:
      axe.set_speed(speed)

for axe in actuator_list:
  axe.set_speed(0)
  axe.close()
