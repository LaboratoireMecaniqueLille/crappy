===================================
More about custom objects in Crappy
===================================

.. role:: py(code)
  :language: python
  :class: highlight

**This last page of the tutorials covers various advanced topics related to**
**the creation of custom objects in Crappy**. Unlike for the three previous
pages, the content of this fourth page will not be of interest for all users.
It is still interesting to go over it for users wanting to have a deeper
understanding of the module, or users with a specific need.

1. Custom Generator Paths
-------------------------

Starting from version 2.0.0, **it is now possible for users to create their**
**own** :ref:`Generator Paths` ! There are two reasons why this possibility was
added so late in the module. First, we're not certain that there is a need for
it. But since only a few modifications were needed to allow the creation of
custom Paths, it was decided to make it possible anyway. And second, the
implementation is a bit messier than for other custom objects. It should still
be accessible for most users though, don't worry !

Just like for the other custom objects, there is a template for creating
custom Paths and the Paths have to be children of
:class:`crappy.blocks.generator_path.meta_path.Path` :

.. code-block:: python

   import crappy

   class MyPath(crappy.blocks.generator_path.meta_path.Path):

       def __init__(self, _last_time, _last_cmd, *_, **__):
           super().__init__(_last_time, _last_cmd)

       def get_cmd(self, data):
           ...

As you can see, there are only two methods to define ! Unlike for the other
custom objects, :meth:`~crappy.blocks.generator_path.meta_path.Path.__init__`
has two mandatory positional arguments that must always be handled and passed
to the parent class for initialization. They represent the last time when a
command was sent by the :ref:`Generator` Block, and the last sent value. In
addition to these two mandatory arguments, users can define as many other
arguments as needed. Note that these two arguments are used for defining the
:py:`t0` and :py:`last_cmd` attributes. that have the same values as
:py:`_last_time` and :py:`_last_cmd` but can be accessed from anywhere in the
Path.

The :meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method is for
generating the next command for the Generator to send. It must return the next
command as a :obj:`float` (:obj:`None` is also acceptable is there's no new
command to send). It accepts one argument, which is the :obj:`dict` returned by
the :meth:`~crappy.blocks.Block.recv_all_data` method of the Generator, and
that contains all the data recently received over incoming Links. It allows to
handle the case when Generator Paths have stop conditions based on the value of
a label, described in :ref:`this tutorials section
<3. Advanced Generator condition>`.

But how to handle the stop conditions ? And how to signal the Generator that a
stop condition was met ? This is where things get a bit trickier ! To indicate
that a stop condition is met, the
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method simply has
to raise a :exc:`StopIteration` exception. That can be done anytime, based on
any arbitrary criterion. However, to make it so that conditions like
:py:`'delay=10'` can be used, a
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method is
provided by the base :class:`~crappy.blocks.generator_path.meta_path.Path`
class. It takes a :obj:`str` or a :obj:`~collections.abc.Callable` or
:obj:`None` as its single argument, and always returns a Callable out of it.
This Callable accepts one argument, which is the :obj:`dict` that is passed as
an argument to :meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd`,
and it returns a :obj:`bool` indicating whether the stop condition is met or
not.

So, to summarize, if your custom Path does not accept a :py:`'condition'` or
equivalent argument, you're free to raise :exc:`StopIteration` whenever you
want to switch to the next Path based on arbitrary criteria. If you do have a
:py:`'condition'` or equivalent argument, you should first parse it during
:meth:`~crappy.blocks.generator_path.meta_path.Path.__init__` using the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method. It
will output a Callable, that you should store as a variable. Then, in the
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method, you should
call this variable with the :obj:`dict` from
:meth:`~crappy.blocks.Block.recv_all_data` as an argument. If it returns
:obj:`True` the condition is met and you should raise :exc:`StopIteration`.
Otherwise, you should return a value for the Generator to send.

It is definitely not the most straightforward implementation, but it is very
flexible and should fit most situations. Let's write a short example to make it
clearer how to create a custom Generator Path and how to handle the
conditions. This example generates a square wave, whose duty cycle can be
either fixed or controlled by the value of an input label :

.. literalinclude:: /downloads/complex_custom_objects/custom_path.py
   :language: python
   :emphasize-lines: 20, 37, 42-43, 51, 54-55

.. Note::
   To run this example, you'll need to have the *matplotlib* and *scipy* Python
   modules installed.

This example contains all the ingredients described above. The parent class is
initialized with the two mandatory arguments, then the :py:`condition` argument
is parsed with
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition`. In
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd`, the given
condition is checked based on the latest received data from upstream Blocks,
and raises :exc:`StopIteration` if needed. This method also returns
:obj:`float` values as expected, and the :py:`t0` attribute is used for
calculating the value to return.

The exact way the custom Path works won't be detailed here, but it should be
self-explanatory by just reading the code and the comments. You can
:download:`download this custom Path example
</downloads/complex_custom_objects/custom_path.py>` to run it locally on your
machine. You should see that the duty cycle of the generated square signal
varies according to the target duty cycle, as expected. In the `examples on
GitHub  <https://github.com/LaboratoireMecaniqueLille/crappy/examples/
custom_objects>`_, you'll find another example of a custom Generator Path.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Path, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.blocks.generator_path.meta_path.Path.log` method provided by
   the parent :class:`~crappy.blocks.generator_path.meta_path.Path` class. This
   way, the log messages are included in the log file and handled in a nicer
   way by Crappy.

There's one more very specific point that we'd like to outline about the use of
Generator Paths in Crappy. Earlier, it was mentioned that the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method of
the base Path object accepts :obj:`~collections.abc.Callable`. More precisely,
it accepts Callables that take as only argument a :obj:`dict` whose keys are
:obj:`str` and values are :obj:`list`, and that return a :obj:`bool` value.
This means that it is actually possible to pass a Callable as the value for
the :py:`condition` argument, not just a :obj:`str` or :obj:`None` ! This
possibility is not often used, but at least you now know that it exists ! It
could for instance come in use if you want to use an existing Path, but you
have an unusual stop condition (e.g. one that depends on the values of two
labels).

2. More about custom InOuts
---------------------------

In addition to what was described in the tutorial section about :ref:`how to
create custom InOut objects <3. Custom InOuts>`, there is one more minor
feature that the :ref:`In / Out` possess and that is worth describing in the
tutorials. That is **the ability for an InOut to acquire data before a test**
**starts, and to use this data to offset the channels to zero**. To do so, the
script must match two conditions. First, the :py:`make_zero_delay` argument of
the :ref:`IOBlock` must be set to a positive value. And second, the used InOut
must have its :meth:`~crappy.inout.InOut.get_data` method defined (it cannot be
a pure stream class). If both of these conditions are met, then the InOut will
acquire data using :meth:`~crappy.inout.InOut.get_data` during
:meth:`~crappy.blocks.IOBlock.prepare` for the specified delay, and create
offsets so that for each acquired channel its value starts from zero at the
beginning of the test. It also works for streams, provided that the number of
channels acquired in *streamer* mode is the same as the number of channels
acquired by :meth:`~crappy.inout.InOut.get_data`.

**Thing get a bit trickier when the hardware can handle and tune offsets for**
**its channels** ! In such a case, it might be advantageous to set the zeroing
offsets directly on the device rather than relying on Crappy. To achieve that,
the :meth:`~crappy.inout.InOut.make_zero` method of the base
:class:`~crappy.inout.InOut` has to be overriden in the child InOut class, and
the way it is performed depends on the capabilities of the hardware. What is
usually done is that the :meth:`~crappy.inout.InOut.make_zero` method of the
base class calculates the offset values, and the one of the child class sets
these values on the hardware and resets the offsets on Crappy's side. This
kind of implementation can be found in the :ref:`Labjack T7` or the
:ref:`Comedi` InOuts. Check their code to see how it looks ! There is also a
very basic example of offsetting in the `examples on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/custom_objects>`_
where the method is overriden and the offsets are simply doubled.

There is no need for a specific example in this sub-section, it is mostly
included to signal the existence of the zeroing feature and the possibility for
users to override it.

3. More about custom Actuators
------------------------------

4. More about custom Cameras
----------------------------

add trigger setting
add software roi
reload

5. More about custom Blocks
---------------------------

niceness, name

6. Including custom objects in a distribution of Crappy
-------------------------------------------------------

You can either add an object locally or to the entire project. If it's locally,
you'll be the only one having access to the modifications but you're free to do
whatever you want. Any modification to the entire project requires an approval
and is subject to few rules, but then everyone will be able to use your object.
**We always recommend you to add any improvement to the entire project, the more
contributions the better !** Here are the different possibilities :

- **Adding your object locally** :

  - If Crappy was installed using ``git``, simply copy a ``.py`` file
    containing your block or your object into the right folder. The class
    inheritance changes compared with an in-script object definition. Refer to
    objects that are already implemented for the appropriate syntax. For example
    if you had :

    .. code-block:: python

       import crappy

       class My_block(crappy.blocks.Block):

    Now you should have :

    .. code-block:: python

       from .block import Block

       class My_block(Block)

    Then modify the ``__init__.py`` file of the folder in which you placed your
    new object. For example if the block mentioned a few lines above is
    contained in ``my_block.py``, you should write in
    ``crappy/blocks/__init__.py`` :

    .. code-block:: python

       from .my_block import My_block

    If you included docstrings in your file and you wish to include them in a
    local documentation, add your object in the corresponding ``.rst`` file in
    the ``/docs/source/crappydocs/`` folder. Again the syntax should be
    self-explanatory. Still following the same example, here we should write in
    ``/docs/source/crappydocs/blocks.rst`` :

    .. code-block:: rst

       My Block
       --------
       .. automodule:: crappy.blocks.my_block
          :members:

    Now simply reinstall Crappy (see :ref:`Installation`, the syntax slightly
    differs according to your OS) and that's it, you can freely use your object
    in scripts !

  - If Crappy was installed using ``pip``,  the quick-and-dirty way is to do
    almost the same steps as in the previous point, except now Crappy's folder
    may be harder to find. If it is installed in a virtualenv you should find it
    easily, otherwise you can open a Python terminal, and type :

      >>> import crappy
      >>> crappy

    This will display the location of Crappy's files. Now like in the previous
    point add your ``.py`` file to the right folder with the right import and
    inheritance modifications, change the corresponding ``__init__.py`` file,
    and that's it ! Next time you import Crappy your object should be available.

      .. Important::
         It's likely that your modifications will be discarded if you then
         update Crappy using ``pip`` !

- **Adding your object to the Crappy project** : see the
  :ref:`Developers information` section. There are a few rules to respect, but
  if your pull request is accepted then all the Crappy users will be able to use
  your object !
