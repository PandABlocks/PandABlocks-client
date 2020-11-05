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
    :meth:`~ControlConnection.receive_bytes` which will return an iterator of
    ``(command, response)`` tuples. The response type will depend on the
    command. For instance `Get` returns `bytes` or a `list` of `bytes` of the
    field value, and `GetFields` returns a `dict` mapping `str` field name to
    `FieldType`.

.. autoclass:: DataConnection
    :noindex:

    The :meth:`~DataConnection.connect` method takes any connection arguments
    and returns the bytes that should be sent to the PandA to make the initial connection.
    Whenever bytes are received from the socket they can be passed to
    :meth:`~DataConnection.receive_bytes` which will return an iterator of
    `Data` objects. Intermediate adjacent `FrameData` instances will be squashed together
    and only emitted when :meth:`~DataConnection.flush` is called.


    The response type will depend on the
    command. For instance `Get` returns `bytes` or a `list` of `bytes` of the
    field value, and `GetFields` returns a `dict` mapping `str` field name to
    `FieldType`.

Wrappers
--------



.. _reference: https://sans-io.readthedocs.io/