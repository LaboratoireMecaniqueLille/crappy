# coding: utf-8

# [data-acquisition-start]
import crappy


def main() -> None:
  acquisition = crappy.blocks.IOBlock(
      'FakeInOut',
      labels=('t(s)', 'ram(%)'),
      freq=2)

  reader = crappy.blocks.LinkReader(name='Memory usage', freq=5)
  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.link(acquisition, reader)
  crappy.link(acquisition, stop)

  crappy.start()


if __name__ == '__main__':
  main()
# [data-acquisition-end]
