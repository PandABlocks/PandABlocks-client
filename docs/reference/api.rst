.. _API:

pandablocks API
===============

The top level pandablocks module contains a number of packages that can be used
from code:

- `pandablocks.commands`: The control commands that can be sent to a PandA
- `pandablocks.responses`: The control and data responses that will be received
- `pandablocks.connections`: Control and data connections that implements the parsing logic
- `pandablocks.asyncio`: An asyncio client that uses the control and data connections
- `pandablocks.blocking`: A blocking client that uses the control and data connections
- `pandablocks.hdf`: Some helpers for efficiently writing data responses to disk


.. automodule:: pandablocks.commands
    :members:

    Commands
    --------

    There is a `Command` subclass for every sort of command that can be sent to
    the `ControlConnection` of a PandA. Many common actions can be accomplished
    with a simple `Get` or `Put`, but some convenience commands like
    `GetBlockNumbers`, `GetFields`, etc. are provided that parse output into
    specific classes.


.. automodule:: pandablocks.responses
    :members:

    Responses
    ---------

    Classes used in responses from both the `ControlConnection` and
    `DataConnection` of a PandA live in this package.

.. automodule:: pandablocks.connections
    :members:

    Connections
    -----------

    `Sans-IO <what_is_sans_io>` connections for both the Control and Data ports
    of PandA TCP server.

.. automodule:: pandablocks.asyncio
    :members:

    Asyncio Client
    --------------

    This is an `asyncio` wrapper to the `ControlConnection` and `DataConnection`
    objects, allowing async calls to ``send(command)`` and iterate over
    ``data()``.

.. automodule:: pandablocks.blocking
    :members:

    Blocking Client
    ---------------

    This is a blocking wrapper to the `ControlConnection` and `DataConnection`
    objects, allowing blocking calls to ``send(commands)`` and iterate over
    ``data()``.

.. automodule:: pandablocks.hdf
    :members:

    HDF Writing
    -----------

    This package contains components needed to write PCAP data to and HDF file
    in the most efficient way. The oneshot `write_hdf_files` is exposed in the
    commandline interface. It assembles a short `Pipeline` of:

        `AsyncioClient` -> `FrameProcessor` -> `HDFWriter`

    The FrameProcessor and HDFWriter run in their own threads as most of the
    heavy lifting is done by `numpy` and ``h5py``, so running in their own threads
    gives multi-CPU benefits without hitting the limit of the GIL.

    The key to the performance of this Pipeline is the use of
    ``data(scaled=False)``. This allows raw data from the TCP server to be used,
    reducing CPU usage of the TCP server on the PandA and allowing about
    55MBytes/s to be produced. The scaling is applied in the client by the
    FrameProcessor before being written to file by the HDFWriter. In tests, this
    data rate could be written to an SSD by a modern multi-core CPU, with
    ``top`` showing a process CPU usage of 60%.

