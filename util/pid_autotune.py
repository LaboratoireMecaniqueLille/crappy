# coding:utf-8

import crappy
from time import sleep, time

"""
This code provides a not-so-stupid method to have an acceptable values for PID 
gains on an output. This will require the ability to send a command to the 
actuator in open loop and this program asserts that it is a non-integrating and 
stable process.

It will give a step setpoint and analyse the response to try to deduce the
gains. Make sure the step size is representative of the utilisation domain (and 
in range) of the actuator.

Warning:
  Make sure to review the constants and to have an idea of how this program 
  will perform before running it !
"""


actuator_kwargs = {"kv": 1200, "inertia": 3, "rv": .5, "fv": 2e-5}
actuator_class = crappy.actuator.Fake_motor
step = 5
loop_delay = .01


def stable(data):
  if len(data) < 100:
    return False
  return all([1.05 * data[-1] > data[i] > .95 * data[-1]
              for i in range(-len(data) // 3, 0)])


graph = crappy.blocks.Grapher(('t(s)', 'speed'))
l = crappy.links.Link()
graph.inputs.append(l)
# graph.start()
graph.launch(time())
act = actuator_class(**actuator_kwargs)
act.open()
v0 = act.get_speed()

act.set_speed(step)  # replace with set_position if needed

d = []
while True:
  speed = act.get_speed()
  print("speed=", speed)
  l.send({'t(s)': time(), 'speed': speed})
  d.append(speed)
  if stable(d):
    break
  sleep(loop_delay)

print("len(data)=", len(d))
v1 = sum(d[-len(d) // 10:]) / (len(d) // 10)
k = (v1 - v0) / step
print("v0=", v0, "v1=", v1, "k=", k)
a0 = sum([v1 - i for i in d]) * loop_delay
tar = a0 / (v1 - v0)
print("A0=", a0, "Tar=", tar)
a1 = sum([i - v0 for i in d[:int(tar / loop_delay)]]) * loop_delay
print("A1=", a1)
t = 2.7 * a1 / (v1 - v0)
print("T=", t)
l = tar - t
print("L=", l)

kp = (.2 + .45 * t / l) / k
ti = (.4 * l + .8 * t) / (l + .1 * t) * l
td = .5 * l * t / (.3 * l + t)

graph.stop()
print("***** GAINS: ******")
print("P=", kp)
print("I=", ti)
print("D=", td)
