# coding: utf-8

from argparse import ArgumentParser
import json
import logging
from pathlib import Path

import crappy

from .scenarios import scenario_builders


def main() -> None:
  """Builds and runs the integration scenario selected on the command line."""

  parser = ArgumentParser(description='Run one Crappy integration scenario.')
  parser.add_argument('scenario', choices=sorted(scenario_builders))
  parser.add_argument('output_dir', type=Path)
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)

  # Keep strong references until crappy.start() returns. Block.instances is a
  # WeakSet, so scenario-local Blocks could otherwise be garbage-collected.
  blocks = scenario_builders[args.scenario](args.output_dir)
  if not blocks:
    raise RuntimeError(f'Scenario {args.scenario!r} did not create any Blocks')

  # Logs are captured by the parent test process and only displayed when the
  # scenario fails, which preserves the child-process traceback for diagnosis.
  crappy.start(log_level=logging.DEBUG)

  with (args.output_dir / 'completed.json').open('w',
                                                  encoding='utf-8') as file:
    json.dump({'scenario': args.scenario}, file)


if __name__ == '__main__':
  main()
