# coding: utf-8

# [modifier-start]
import crappy


def main() -> None:
  speed = crappy.blocks.Generator(
      path=({'type': 'Constant',
             'value': 2,
             'condition': 'delay=4'},),
      cmd_label='speed(mm/s)',
      spam=True,
      freq=5,
      end_delay=0.2)

  reader = crappy.blocks.LinkReader(
      name='Speed and calculated position',
      freq=5)

  integrate_speed = crappy.modifier.Integrate(
      label='speed(mm/s)',
      out_label='position(mm)')

  crappy.link(speed, reader, modifier=integrate_speed)

  crappy.start()


if __name__ == '__main__':
  main()
# [modifier-end]
