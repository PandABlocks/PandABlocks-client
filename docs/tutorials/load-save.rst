.. _load-save:

Commandline Load/Save Tutorial
==============================

This tutorial shows how to use the commandline tool to save
the state of all the Blocks and Fields in a PandA, and load a new state from file. It
assumes that you know the basic concepts of a PandA as outlined in the PandABlocks-FPGA
blinking LEDs tutorial_.

Save
----

Port save-state from TCP server repo, use this to save current state. Show what
the file might look like

Load
----

Port load-state from TCP server repo to cli args, and
document loading a canned demo state for the next tutorial. Add
a ``--tutorial`` flag that loads this canned state that should be distributed
with the module.

Web GUI
-------

Add a web gui screenshot that demonstrates this


.. _tutorial: https://pandablocks-fpga.readthedocs.io/en/latest/tutorials/tutorial1_blinking_leds.html