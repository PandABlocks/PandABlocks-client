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

.. automodule:: pandablocks.responses
    :members:

.. automodule:: pandablocks.connections
    :members:

.. automodule:: pandablocks.asyncio
    :members:

.. automodule:: pandablocks.blocking
    :members:

.. automodule:: pandablocks.hdf
    :members:
