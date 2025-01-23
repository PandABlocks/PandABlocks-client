[![CI](https://github.com/PandABlocks/PandABlocks-client/actions/workflows/ci.yml/badge.svg)](https://github.com/PandABlocks/PandABlocks-client/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/PandABlocks/PandABlocks-client/branch/main/graph/badge.svg)](https://codecov.io/gh/PandABlocks/PandABlocks-client)
[![PyPI](https://img.shields.io/pypi/v/pandablocks.svg)](https://pypi.org/project/pandablocks)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# pandablocks

A Python client to control and data ports of the PandABlocks TCP server

Source          | <https://github.com/PandABlocks/PandABlocks-client>
:---:           | :---:
PyPI            | `pip install pandablocks`
Docker          | `docker run ghcr.io/pandablocks/PandABlocks-client:latest`
Documentation   | <https://pandablocks.github.io/PandABlocks-client>
Releases        | <https://github.com/PandABlocks/PandABlocks-client/releases>

Command line tool features an interactive console, load/save control, and HDF5 writing:

```shell
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
```

Library features a Sans-IO core with both asyncio and blocking wrappers:

```python
from pandablocks.blocking import BlockingClient
from pandablocks.commands import Get

with BlockingClient("hostname-or-ip") as client:
    # Commands sent to Control port
    idn = client.send(Get("*IDN"))
    print(f"Hello {idn}")
    for data in client.data():
        # Data captured from Data port
        print(f"I got some PCAP data {data}")
```

<!-- README only content. Anything below this line won't be included in index.md -->

See https://pandablocks.github.io/PandABlocks-client for more detailed documentation.
