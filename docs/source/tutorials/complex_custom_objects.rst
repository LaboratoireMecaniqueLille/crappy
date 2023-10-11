===================================
More about custom objects in Crappy
===================================

1. Custom Generator Paths
-------------------------

2. More about custom InOuts
---------------------------

Make zero
Streams

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
