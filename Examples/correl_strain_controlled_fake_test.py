# coding: utf-8

"""
Demonstration of a DIC controlled test.

This program is intended as a demonstration and is fully virtual. The strain is
measured using DIC and a PID controller drives the machine to apply a ramp of
strain.

No hardware required.
Requires the cv2 module to be installed.
"""

import crappy

if __name__ == "__main__":
  img = crappy.resources.speckle

  speed = .05  # Strain rate (%/s)

  # Simply create a sawtooth signal at 0.05 units/s between 0 and 1
  # Note that this generator takes no feedback
  generator = crappy.blocks.Generator(path=[
    {'type': 'cyclic_ramp', 'speed1': speed, 'condition1': 'target_Exx(%)>1',
     'speed2': -speed, 'condition2': 'target_Exx(%)<0', 'cycles': 100}, ],
                                      cmd_label='target_Exx(%)')

  # Our fake machine
  machine = crappy.blocks.Fake_machine(maxstrain=1.7, k=5000, l0=20,
                                       plastic_law=lambda exx: 0,
                                       sigma={'F(N)': 0.5},
                                       cmd_label='pid')

  # The block performing the DIC
  dis = crappy.blocks.DISCorrel('', input_label='frame', show_image=True,
                                labels=['t(s)', 'x', 'y', 'measured_Exx(%)',
                                        'measured_Eyy(%)'],
                                verbose=True, iterations=0, finest_scale=2)

  # This modifier will generate an image with the values of strain
  # coming from the Fake_machine block
  crappy.link(machine, dis, modifier=crappy.modifier.Apply_strain_img(img))

  # The PID block takes TWO inputs: the setpoint and the feedback
  # Here the setpoint is coming from the generator (target_Exx(%))
  # and the feedback is the strain measured with the DIC (measured_Exx(%))
  # The output is the command sent to the machine
  # In a real-world scenario, consider using out_min and out_max
  # to clamp the output value and i_limit to prevent over_integration.
  pid = crappy.blocks.PID(kp=0.5, ki=2, kd=0.05, target_label='target_Exx(%)',
                          input_label='measured_Exx(%)', send_terms=True)

  # We link the two inputs and the output
  # The order does not matter, the inputs are identified thanks to the
  # keywords target_label (setpoint) and input_label (feedback)
  crappy.link(generator, pid)
  crappy.link(dis, pid)
  crappy.link(pid, machine)

  # To see what is sent to the machine
  # Since send_terms is given to the PID block, it also returns the 3
  # channels of the PID (P, I and D). This can be useful when adjusting
  # the gains
  graph_pid = crappy.blocks.Grapher(('t(s)', 'pid'),
                                    ('t(s)', 'p_term'), ('t(s)', 'i_term'),
                                    ('t(s)', 'd_term'))
  crappy.link(pid, graph_pid)

  # To see the commanded and the measured strains
  graph_def2 = crappy.blocks.Grapher(('t(s)', 'measured_Exx(%)'),
                                     ('t(s)', 'target_Exx(%)'))
  crappy.link(dis, graph_def2)
  crappy.link(generator, graph_def2)

  crappy.start()
