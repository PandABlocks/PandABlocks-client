PandABlocks Python Client
=========================

|build_status| |coverage| |pypi_version| |license|

A Python client to control and data ports of the PandABlocks TCP server.

============== ==============================================================
PyPI           ``pip install pandablocks``
Source code    https://github.com/PandABlocks/PandABlocks-client
Documentation  https://pandablocks.github.io/PandABlocks-client/master/index.html
============== ==============================================================

TODO - Test 0.1.4

Command line tool features an interactive console, load/save control, and HDF5
writing:

.. code::

    $ pip install pandablocks

    $ pandablocks control <panda-hostname>
    < PCAP.     # Hit TAB key...
    PCAP.ACTIVE     PCAP.BITS1      PCAP.BITS3      PCAP.GATE       PCAP.SAMPLES    PCAP.TRIG       PCAP.TS_END     PCAP.TS_TRIG
    PCAP.BITS0      PCAP.BITS2      PCAP.ENABLE     PCAP.HEALTH     PCAP.SHIFT_SUM  PCAP.TRIG_EDGE  PCAP.TS_START
    < PCAP.ACTIVE?
    OK =1

    $ pandablocks hdf <panda-hostname> /tmp/panda-%d.h5
    INFO:Opened '/tmp/panda-1.h5' with 60 byte samples stored in 11 datasets
    INFO:Closed '/tmp/panda-1.h5' after writing 50000000 samples. End reason is 'Disarmed'

Library features a Sans-IO core with both asyncio and blocking wrappers:

.. code:: python

    from pandablocks.blocking import BlockingClient
    from pandablocks.commands import Get

    with BlockingClient("hostname-or-ip") as client:
        # Commands sent to Control port
        idn = client.send(Get("*IDN"))
        print(f"Hello {idn}")
        for data in client.data():
            # Data captured from Data port
            print(f"I got some PCAP data {data}")



.. |build_status| image:: https://github.com/PandABlocks/PandABlocks-client/workflows/Python%20CI/badge.svg?branch=master
    :target: https://github.com/PandABlocks/PandABlocks-client/actions?query=workflow%3A%22Python+CI%22
    :alt: Build Status

.. |coverage| image:: https://pandablocks.github.io/PandABlocks-client/coverage.svg
    :target: https://github.com/PandABlocks/PandABlocks-client/actions?query=workflow%3A%22Python+CI%22
    :alt: Test Coverage

.. |pypi_version| image:: https://badge.fury.io/py/pandablocks.svg
    :target: https://badge.fury.io/py/pandablocks
    :alt: Latest PyPI version

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache License