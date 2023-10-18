.. _API:

API
===

The top level pandablocks module contains a number of packages that can be used
from code:

- `pandablocks.commands`: The control commands that can be sent to a PandA
- `pandablocks.responses`: The control and data responses that will be received
- `pandablocks.connections`: Control and data connections that implements the parsing logic
- `pandablocks.asyncio`: An asyncio client that uses the control and data connections
- `pandablocks.blocking`: A blocking client that uses the control and data connections
- `pandablocks.hdf`: Some helpers for efficiently writing data responses to disk
- `pandablocks.utils`: General utility methods for use with pandablocks


.. automodule:: pandablocks.commands
    :members:

    Commands
    --------

    There is a `Command` subclass for every sort of command that can be sent to
    the `ControlConnection` of a PandA. Many common actions can be accomplished
    with a simple `Get` or `Put`, but some convenience commands like
    `GetBlockInfo`, `GetFieldInfo`, etc. are provided that parse output into
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

    `Sans-IO <sans-io>` connections for both the Control and Data ports
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
    heavy lifting is done by numpy_ and h5py_, so running in their own threads
    gives multi-CPU benefits without hitting the limit of the GIL.

    .. seealso:: `library-hdf`, `performance`

.. automodule:: pandablocks.utils    
    
    Utilities
    ---------

    This package contains general methods for working with pandablocks.
