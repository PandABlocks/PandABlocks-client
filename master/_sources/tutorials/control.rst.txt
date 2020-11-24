Interactive Control Tutorial
============================

This tutorial shows how to use the commandline tool to open an interactive terminal
to control a PandA.

Connect
-------

Open a terminal, and type::

    pandablocks control <hostname>

Where ``<hostname>`` is the hostname or ip address of your PandA.

Type Commands
-------------

You should be presented with a prompt where you can type PandABlocks-server
commands_. If you are on Linux you can tab complete commands with the TAB key::

    < PCAP.     # Hit TAB key...
    PCAP.ACTIVE     PCAP.BITS1      PCAP.BITS3      PCAP.GATE       PCAP.SAMPLES    PCAP.TRIG       PCAP.TS_END     PCAP.TS_TRIG
    PCAP.BITS0      PCAP.BITS2      PCAP.ENABLE     PCAP.HEALTH     PCAP.SHIFT_SUM  PCAP.TRIG_EDGE  PCAP.TS_START

Pressing return will send the command to the server and display the response.

Control an acquisition
----------------------

You can check if an acquisition is currently in progress by getting the value of the
``PCAP.ACTIVE`` field::

    < PCAP.ACTIVE?
    OK =0

You can start and stop acquisitions with special "star" commands. To start an acquisition::

    < *PCAP.ARM=
    OK

You can now use the up arrow to recall the previous command, then press return::

    < PCAP.ACTIVE?
    OK =1

This means that acquisition is in progress. You can stop it by disarming::

    < *PCAP.DISARM=
    OK
    < PCAP.ACTIVE?
    OK =0

Conclusion
----------

This tutorial has shown how to start and stop an acquisition from the commandline
client. It can also be used to send any other control commands_ to query and set
variables on the PandA.

.. _commands: https://pandablocks-server.readthedocs.io/en/latest/commands.html
