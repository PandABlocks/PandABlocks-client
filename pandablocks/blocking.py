import socket
from typing import Any, Iterator, List, Tuple, Union, overload

from .core import Command, ControlConnection, Data, DataConnection, T


class BlockingClient:
    def __init__(self, host: str):
        self._host = host
        self._ctrl_socket = self._make_socket(8888)
        self._ctrl_connection = ControlConnection()

    def _make_socket(self, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._host, port))
        s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        return s

    def close(self, s: socket.socket = None):
        if s is None:
            s = self._ctrl_socket
        s.shutdown(socket.SHUT_WR)
        s.close()

    @overload
    def send(self, commands: Command[T]) -> T:
        ...

    @overload
    def send(self, commands: List[Command]) -> List:
        ...

    def send(self, commands: Union[Command[T], List[Command]]):
        if isinstance(commands, Command):
            commands = [commands]
        for command in commands:
            bytes = self._ctrl_connection.send(command)
            self._ctrl_socket.sendall(bytes)
        responses: List[Tuple[Command, Any]] = []
        while len(responses) < len(commands):
            bytes = self._ctrl_socket.recv(4096)
            responses += list(self._ctrl_connection.receive_data(bytes))
        assert all(
            c == r[0] for c, r in zip(commands, responses)
        ), f"Mismatched {commands} and {responses}"
        if len(responses) == 1:
            return responses[0][1]
        else:
            return [r[1] for r in responses]

    def data(self, frame_timeout: int = None) -> Iterator[Data]:
        s = self._make_socket(8889)
        s.settimeout(frame_timeout)  # close enough
        try:
            connection = DataConnection()
            s.sendall(connection.connect())
            while True:
                bytes = s.recv(4096)
                for data in connection.receive_data(bytes):
                    yield data
        finally:
            self.close(s)
