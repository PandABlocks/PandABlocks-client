.. _sans-io:

Why write a Sans-IO library?
============================

As the reference_ says: *Reusability*. The protocol can be coded in a separate
class to the I/O allowing integration into a number of different concurrency
frameworks.

For instance, we need both a `BlockingClient` and an `AsyncioClient`. If we had
coded the protocol in either of them it would not be usable in the other. Much
better to put it in a separate class and feed it bytes off the wire. We call
this protocol encapsulation a Connection.

Connections
-----------

The PandA TCP server exposes a Control port and a Data port, so there are
corresponding `ControlConnection` and `DataConnection` objects:

.. currentmodule:: pandablocks.connections

.. autoclass:: ControlConnection
    :noindex:

    The :meth:`~ControlConnection.send` method takes a `Command` subclass and
    returns the bytes that should be sent to the PandA. Whenever bytes are
    received from the socket they can be passed to
    :meth:`~ControlConnection.receive_bytes` which will return any subsequent
    bytes that should be send back. The :meth:`~ControlConnection.responses`
    method returns an iterator of ``(command, response)`` tuples that have now
    completed. The response type will depend on the command. For instance `Get`
    returns `bytes` or a `list` of `bytes` of the field value, and `GetFieldInfo`
    returns a `dict` mapping `str` field name to `FieldInfo`.

.. autoclass:: DataConnection
    :noindex:

    The :meth:`~DataConnection.connect` method takes any connection arguments
    and returns the bytes that should be sent to the PandA to make the initial
    connection. Whenever bytes are received from the socket they can be passed
    to :meth:`~DataConnection.receive_bytes` which will return an iterator of
    `Data` objects. Intermediate `FrameData` can be squashed together by passing
    ``flush_every_frame=False``, then explicitly calling
    :meth:`~DataConnection.flush` when they are required.

Wrappers
--------

Of course, these Connections are useless without connecting some I/O. To aid with
this, wrappers are included for use in `asyncio <asyncio>` and blocking programs. They expose
slightly different APIs to make best use of the features of their respective concurrency frameworks.

For example, to send multiple commands in fields with the `blocking` wrapper::

    with BlockingClient("hostname") as client:
        resp1, resp2 = client.send([cmd1, cmd2])

while with the `asyncio` wrapper we would::

    async with AsyncioClient("hostname") as client:
        resp1, resp2 = await asyncio.gather(
            client.send(cmd1),
            client.send(cmd2)
        )

The first has the advantage of simplicity, but blocks while waiting for data.
The second allows multiple co-routines to use the client at the same time at the
expense of a more verbose API.

The wrappers do not guarantee feature parity, for instance the ``flush_period``
option is only available in the asyncio wrapper.







.. _reference: https://sans-io.readthedocs.io/