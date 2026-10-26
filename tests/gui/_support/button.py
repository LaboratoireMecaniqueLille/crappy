from tests.support.blocks import with_failure_manager
from multiprocessing import Value
from unittest.mock import patch
from crappy.blocks.button import Button
import crappy.blocks.button as button_module
from tests.support.blocks import TestBlock, link


class ButtonTests:
  """Run the same signal and counter checks against both GUI backends."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Tracks the GUI Blocks to destroy them during teardown."""

    super().setUp()
    patcher = patch.object(button_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._buttons: list[Button] = list()

  def tearDown(self) -> None:
    """Closes windows before resetting the Block class state."""

    for button in self._buttons:
      button.finish()

    super().tearDown()

  def _prepare_button(self, **kwargs) -> tuple[Button, TestBlock]:
    """Creates a linked Button and prepares its GUI."""

    kwargs.setdefault('backend', self._backend)
    button = with_failure_manager(Button(**kwargs))
    button._instance_t0 = Value('d', self._t0)
    sink = TestBlock()
    link(button, sink)

    self._buttons.append(button)
    button.prepare()
    if self._backend == 'tkinter':
      button._root.withdraw()
    return button, sink

  def _text(self, button: Button) -> str:
    """Read the visible counter without coupling common checks to Tk."""

    return (button._text.get() if self._backend == 'tkinter'
            else button._qt_label.text())

  def _click(self, button: Button) -> None:
    """Invoke the actual widget callback for the selected backend."""

    if self._backend == 'tkinter':
      button._button.invoke()
    else:
      button._qt_button.click()

  def test_label_arguments_are_validated(self) -> None:
    """Checks that invalid labels are rejected early."""

    with self.assertRaises(TypeError):
      with_failure_manager(Button(time_label=1))

    with self.assertRaises(TypeError):
      with_failure_manager(Button(label=1))

    with self.assertRaises(ValueError):
      with_failure_manager(Button(time_label='same', label='same'))

    self.assertEqual(Button(time_label='time', label='trigger').labels,
                     ['time', 'trigger'])

  def test_prepare_requires_output_link(self) -> None:
    """Checks that a Button without output Links fails early."""

    button = with_failure_manager(Button())

    with self.assertRaises(IOError):
      button.prepare()

  def test_prepare_creates_gui_with_custom_label(self) -> None:
    """Checks the initial window and native widgets created by prepare."""

    button, _ = self._prepare_button(time_label='time', label='trigger')

    self.assertEqual(button._step, 0)
    self.assertEqual(self._text(button), 'trigger: 0')
    if self._backend == 'tkinter':
      self.assertEqual(button._root.title(), 'Button block')
      self.assertEqual(str(button._label.cget('textvariable')),
                       str(button._text))
      self.assertEqual(button._button.cget('text'), 'Next step')
      self.assertIsNone(button._qt_window)
    else:
      self.assertEqual(button._qt_window.windowTitle(), 'Button block')
      self.assertEqual(button._qt_button.text(), 'Next step')
      self.assertIsNone(button._root)
      layout = button._qt_window.layout()
      self.assertIs(layout.itemAt(0).widget(), button._qt_label)
      self.assertIs(layout.itemAt(1).layout().itemAt(0).widget(),
                     button._qt_button)
      constraints = button_module.QtWidgets.QLayout.SizeConstraint
      self.assertEqual(layout.sizeConstraint(),
                       constraints.SetFixedSize)

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
      self._click(button)

    self.assertEqual(button._step, 1)
    self.assertEqual(self._text(button), 'trigger: 1')
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

  def test_finish_destroys_window(self) -> None:
    """Checks that repeated finish calls close the Block's window."""

    button, _ = self._prepare_button()
    window = (button._root if self._backend == 'tkinter'
              else button._qt_window)

    button.finish()
    button.finish()

    if self._backend == 'tkinter':
      from tkinter import TclError
      self.assertIsNone(button._root)
      with self.assertRaises(TclError):
        window.wm_state()
    else:
      self.assertIsNone(button._qt_window)
      self.assertFalse(window.isVisible())
