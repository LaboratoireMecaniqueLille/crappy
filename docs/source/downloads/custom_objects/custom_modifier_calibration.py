# coding: utf-8

# [custom-modifier-start]
import crappy


# [custom-modifier-class-start]
class VoltageToForce(crappy.modifier.Modifier):

  def __init__(self, sensitivity: float, zero: float) -> None:
    super().__init__()
    self._sensitivity = sensitivity
    self._zero = zero

  def __call__(self, data: dict) -> dict:
    data['force(N)'] = ((data['voltage(V)'] - self._zero) *
                        self._sensitivity)
    return data
# [custom-modifier-class-end]


def main() -> None:
  voltage = crappy.blocks.Generator(
      path=({'type': 'Constant',
             'value': 0.25,
             'condition': 'delay=3'},),
      cmd_label='voltage(V)',
      spam=True,
      freq=5,
      end_delay=0.2)

  reader = crappy.blocks.LinkReader(name='Calibrated load cell', freq=5)

  # [custom-modifier-use-start]
  calibration = VoltageToForce(sensitivity=100, zero=0.02)
  crappy.link(voltage, reader, modifier=calibration)
  # [custom-modifier-use-end]

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-modifier-end]
