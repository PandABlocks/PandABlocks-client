PandABlocks Python Client
=========================

|build_status| |coverage| |pypi_version| |readthedocs|

A Python client to control and data ports of the PandABlocks TCP server.

Library features a Sans-IO core with both asyncio and blocking wrappers:

.. code:: python

    from pandablocks.blocking import BlockingClient
    from pandablocks.commands import Get

    client = BlockingClient("<panda-hostname>")
    idn = client.send(Get("*IDN"))
    print(f"Hello {idn}")

    for data in client.data():
        print(f"I got some PCAP data {data}")

    client.close()

Command line tool features an interactive console, load/save control, and HDF5
writing:

.. code:: python

    $ pandablocks control <panda-hostname>
    < PCAP.     # Hit TAB key...
    PCAP.ACTIVE     PCAP.BITS1      PCAP.BITS3      PCAP.GATE       PCAP.SAMPLES    PCAP.TRIG       PCAP.TS_END     PCAP.TS_TRIG
    PCAP.BITS0      PCAP.BITS2      PCAP.ENABLE     PCAP.HEALTH     PCAP.SHIFT_SUM  PCAP.TRIG_EDGE  PCAP.TS_START
    < PCAP.ACTIVE?
    OK =1

    $ pandablocks hdf <panda-hostname> /tmp/panda-%d.h5
    INFO:Wrote 50000000 samples into /tmp/panda-1.h5, end reason 'Disarmed'


Documentation
-------------

Full documentation is available at https://pandablocks-client.readthedocs.io


Source Code
-----------

Available from https://github.com/PandABlocks/PandABlocks-client


Contributing
------------

See `CONTRIBUTING`_


License
-------
APACHE License. (see `LICENSE`_)


.. |build_status| image:: https://travis-ci.com/PandABlocks/PandABlocks-client.svg?branch=master
    :target: https://travis-ci.com/PandABlocks/PandABlocks-client
    :alt: Build Status

.. |coverage| image:: https://coveralls.io/repos/github/PandABlocks/PandABlocks-client/badge.svg?branch=master
    :target: https://coveralls.io/github/PandABlocks/PandABlocks-client?branch=master
    :alt: Test Coverage

.. |pypi_version| image:: https://badge.fury.io/py/pandablocks.svg
    :target: https://badge.fury.io/py/pandablocks
    :alt: Latest PyPI version

.. |readthedocs| image:: https://readthedocs.org/projects/pandablocks-client/badge/?version=latest
    :target: https://pandablocks-client.readthedocs.io
    :alt: Documentation

.. _CONTRIBUTING:
    https://github.com/PandABlocks/PandABlocks-client/blob/master/CONTRIBUTING.rst

.. _LICENSE:
    https://github.com/PandABlocks/PandABlocks-client/blob/master/LICENSE
