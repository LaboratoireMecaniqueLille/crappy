# coding: utf-8

from multiprocessing import Value
from tkinter import TclError
from unittest.mock import patch
from crappy.blocks.button import Button
import crappy.blocks.button as button_module

from ..block import BlockTestBase, TestBlock, link


class TestButton(BlockTestBase):
  """Unit tests for the Button Block-specific GUI behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Tracks the GUI Blocks to destroy them during teardown."""

    self._buttons: list[Button] = list()

  def tearDown(self) -> None:
    """Destroys Tk windows before resetting the Block class state."""

    for button in self._buttons:
      button.finish()

    super().tearDown()

  def _prepare_button(self, **kwargs) -> tuple[Button, TestBlock]:
    """Creates a linked Button and prepares its GUI."""

    button = Button(**kwargs)
    button._instance_t0 = Value('d', self._t0)
    sink = TestBlock()
    link(button, sink)

    button.prepare()
    button._root.withdraw()

    self._buttons.append(button)
    return button, sink

  def test_label_arguments_are_validated(self) -> None:
    """Checks that invalid labels are rejected early."""

    with self.assertRaises(TypeError):
      Button(time_label=1)

    with self.assertRaises(TypeError):
      Button(label=1)

    with self.assertRaises(ValueError):
      Button(time_label='same', label='same')

    self.assertEqual(Button(time_label='time', label='trigger').labels,
                     ['time', 'trigger'])

  def test_prepare_requires_output_link(self) -> None:
    """Checks that a Button without output Links fails early."""

    button = Button()

    with self.assertRaises(IOError):
      button.prepare()

  def test_prepare_creates_gui_with_custom_label(self) -> None:
    """Checks the initial Tk state created by prepare."""

    button, _ = self._prepare_button(time_label='time', label='trigger')

    self.assertEqual(button._root.title(), 'Button block')
    self.assertEqual(button._step.get(), 0)
    self.assertEqual(button._text.get(), 'trigger: 0')
    self.assertEqual(str(button._label.cget('textvariable')),
                     str(button._text))
    self.assertEqual(button._button.cget('text'), 'Next step')

  def test_begin_sends_initial_zero_when_requested(self) -> None:
    """Checks the optional initial zero emitted at Block start."""

    button, sink = self._prepare_button(send_0=True,
                                        time_label='time',
                                        label='trigger')

    with patch.object(button_module, 'time', return_value=12.5):
      button.begin()

    self.assertEqual(sink.inputs[0].recv(), {'time': 2.5, 'trigger': 0})

  def test_begin_does_not_send_initial_zero_by_default(self) -> None:
    """Checks that the default begin call stays quiet."""

    button, sink = self._prepare_button(send_0=False, spam=False)

    button.begin()

    self.assertFalse(sink.inputs[0].poll())

  def test_begin_sends_initial_zero_in_spam_mode(self) -> None:
    """Checks that spam mode initializes downstream Blocks with step zero."""

    button, sink = self._prepare_button(spam=True,
                                        time_label='time',
                                        label='trigger')

    with patch.object(button_module, 'time', return_value=12.5):
      button.begin()

    self.assertEqual(sink.inputs[0].recv(), {'time': 2.5, 'trigger': 0})

  def test_button_click_updates_step_text_and_sends_payload(self) -> None:
    """Checks the click callback state update and emitted message."""

    button, sink = self._prepare_button(time_label='time', label='trigger')

    with patch.object(button_module, 'time', return_value=13.0):
      button._button.invoke()

    self.assertEqual(button._step.get(), 1)
    self.assertEqual(button._text.get(), 'trigger: 1')
    self.assertEqual(sink.inputs[0].recv(), {'time': 3.0, 'trigger': 1})

  def test_loop_only_sends_in_spam_mode(self) -> None:
    """Checks loop payload emission for regular and spam modes."""

    button, sink = self._prepare_button(spam=False)

    button.loop()

    self.assertFalse(sink.inputs[0].poll())

    spam_button, spam_sink = self._prepare_button(spam=True,
                                                  time_label='time',
                                                  label='trigger')

    with patch.object(button_module, 'time', return_value=14.0):
      spam_button.loop()

    self.assertEqual(spam_sink.inputs[0].recv(), {'time': 4.0, 'trigger': 0})

  def test_loop_ignores_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    button, sink = self._prepare_button(spam=True)

    with patch.object(button._root, 'update', side_effect=TclError):
      button.loop()

    self.assertFalse(sink.inputs[0].poll())

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes the Tk root."""

    button, _ = self._prepare_button()

    button.finish()

    with self.assertRaises(TclError):
      button._root.wm_state()
