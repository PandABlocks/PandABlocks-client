import socket
from typing import Iterable, Iterator, List, Optional, Union, overload

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data

# Define the public API of this module
__all__ = ["BlockingClient"]


class _SocketHelper:
    _socket: Optional[socket.socket] = None

    def connect(self, host: str, port: int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self._socket = s

    @property
    def socket(self) -> socket.socket:
        assert self._socket, "connect() not called yet"
        return self._socket

    def close(self):
        s = self.socket
        self._socket = None
        s.shutdown(socket.SHUT_WR)
        s.close()


class BlockingClient:
    """Blocking implementation of a PandABlocks client.
    For example::

        with BlockingClient("hostname-or-ip") as client:
            # Control port is now connected
            resp1, resp2 = client.send([cmd1, cmd2])
            resp3 = client.send(cmd3)
            for data in client.data():
                handle(data)
        # Control and data ports are now disconnected
    """

    def __init__(self, host: str):
        self._host = host
        self._ctrl_connection = ControlConnection()
        self._ctrl_socket = _SocketHelper()

    def connect(self):
        """Connect to the control port, and be ready to handle commands"""
        self._ctrl_socket.connect(self._host, 8888)

    def close(self):
        """Close the control connection, and wait for completion"""
        self._ctrl_socket.close()

    def __enter__(self) -> "BlockingClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @overload
    def send(self, commands: Command[T], timeout: Optional[int] = None) -> T:
        ...

    @overload
    def send(self, commands: Iterable[Command], timeout: Optional[int] = None) -> List:
        ...

    def send(
        self,
        commands: Union[Command[T], Iterable[Command]],
        timeout: Optional[int] = None,
    ):
        """Send a command to control port of the PandA, returning its response.

        Args:
            commands: If single `Command`, return its response. If a list of commands
                return a list of reponses
            timeout: If no reponse in this time, raise `socket.timeout`
        """
        s = self._ctrl_socket.socket
        s.settimeout(timeout)
        if isinstance(commands, Command):
            commands = [commands]
        else:
            commands = list(commands)
        for command in commands:
            to_send = self._ctrl_connection.send(command)
            s.sendall(to_send)
        # Rely on dicts being ordered, Ellipsis is shorthand for "no response yet"
        cr = {id(command): ... for command in commands}
        while ... in cr.values():
            received = s.recv(4096)
            to_send = self._ctrl_connection.receive_bytes(received)
            s.sendall(to_send)
            for command, response in self._ctrl_connection.responses():
                assert cr[id(command)] is ..., "Already got response for {command}"
                cr[id(command)] = response
        responses = list(cr.values())
        for response in responses:
            if isinstance(response, Exception):
                raise response
        if len(responses) == 1:
            return responses[0]
        else:
            return responses

    def data(
        self, scaled: bool = True, frame_timeout: Optional[int] = None
    ) -> Iterator[Data]:
        """Connect to data port and yield data frames

        Args:
            scaled: Whether to scale and average data frames, reduces throughput
            frame_timeout: If no data is received for this amount of time, raise
                `socket.timeout`
        """
        data_socket = _SocketHelper()
        data_socket.connect(self._host, 8889)

        connection = DataConnection()
        s = data_socket.socket
        s.settimeout(frame_timeout)  # close enough
        s.sendall(connection.connect(scaled))
        try:
            while True:
                received = s.recv(4096)
                yield from connection.receive_bytes(received)
        finally:
            data_socket.close()
