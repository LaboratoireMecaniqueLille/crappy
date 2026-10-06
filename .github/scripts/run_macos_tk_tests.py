# coding: utf-8

"""Run Tk GUI tests while preserving macOS's cached first Tcl interpreter."""

from pathlib import Path
import runpy
import sys


_retained_interpreters = []


def main() -> None:
  """Keep the first interpreter alive and pass CLI arguments to unittest."""

  import tkinter as tk

  tk_init = tk.Tk.__init__

  def retain_first_interpreter(self, *args, **kwargs) -> None:
    """Retain the interpreter, not its root, Block, or test case."""

    tk_init(self, *args, **kwargs)
    if not _retained_interpreters:
      _retained_interpreters.append(self.tk)

  # Cocoa caches the first Tcl interpreter for native menu callbacks. Keep it
  # alive until process exit, even after its window has been destroyed
  tk.Tk.__init__ = retain_first_interpreter

  # Script execution puts .github/scripts, rather than the repository root,
  # on sys.path. Make the repository's tests importable, as with python -m
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  # Preserve the normal unittest entry point for multiprocessing spawn
  runpy.run_module('unittest', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
  main()
