import socket
from typing import Any, Iterable, Iterator, List, Tuple, Union, overload

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data


class BlockingClient:
    """Blocking implementation of a PandABlocks client.
    For example::

        # Control port is connected during this call
        client = BlockingClient("hostname-or-ip")
        resp1, resp2 = client.send([cmd1, cmd2])
        for data in client.data():
            handle(data)
        client.close_control()
    """

    def __init__(self, host: str):
        self._host = host
        self._ctrl_socket = self._make_socket(8888)
        self._ctrl_connection = ControlConnection()

    def _make_socket(self, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._host, port))
        s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        return s

    def close_control(self, s: socket.socket = None):
        """Close control connection and wait for completion"""
        if s is None:
            s = self._ctrl_socket
        s.shutdown(socket.SHUT_WR)
        s.close()

    @overload
    def send(self, commands: Command[T], timeout: int = None) -> T:
        ...

    @overload
    def send(self, commands: Iterable[Command], timeout: int = None) -> List:
        ...

    def send(self, commands: Union[Command[T], Iterable[Command]], timeout: int = None):
        """Send a command to control port of the PandA, returning its response.

        Args:
            commands: If single `Command`, return its response. If a list of commands
                return a list of reponses
            timeout: If no reponse in this time, raise `socket.timeout`
        """
        self._ctrl_socket.settimeout(timeout)
        if isinstance(commands, Command):
            commands = [commands]
        else:
            commands = list(commands)
        for command in commands:
            bytes = self._ctrl_connection.send(command)
            self._ctrl_socket.sendall(bytes)
        responses: List[Tuple[Command, Any]] = []
        while len(responses) < len(commands):
            bytes = self._ctrl_socket.recv(4096)
            responses += list(self._ctrl_connection.receive_bytes(bytes))
        assert all(
            c == r[0] for c, r in zip(commands, responses)
        ), f"Mismatched {commands} and {responses}"
        if len(responses) == 1:
            return responses[0][1]
        else:
            return [r[1] for r in responses]

    def data(self, scaled: bool = True, frame_timeout: int = None) -> Iterator[Data]:
        """Connect to data port and yield data frames

        Args:
            scaled: Whether to scale and average data frames, reduces throughput
            frame_timeout: If no data is received for this amount of time, raise
                `socket.timeout`
        """
        s = self._make_socket(8889)
        s.settimeout(frame_timeout)  # close enough
        try:
            connection = DataConnection()
            s.sendall(connection.connect(scaled))
            # bool(True) instead of True so IDE sees finally block is reachable
            while bool(True):
                bytes = s.recv(4096)
                yield from connection.receive_bytes(bytes)
                yield from connection.flush()
        finally:
            self.close_control(s)
