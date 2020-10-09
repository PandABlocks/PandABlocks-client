import socket
from typing import Any, Iterable, Iterator, List, Tuple, Union, overload

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data


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
    def send(self, commands: Command[T], timeout: int = None) -> T:
        ...

    @overload
    def send(self, commands: Iterable[Command], timeout: int = None) -> List:
        ...

    def send(self, commands: Union[Command[T], Iterable[Command]], timeout: int = None):
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
        s = self._make_socket(8889)
        s.settimeout(frame_timeout)  # close enough
        try:
            connection = DataConnection()
            s.sendall(connection.connect(scaled))
            # bool(True) instead of True so IDE sees finally block is reachable
            while bool(True):
                bytes = s.recv(4096)
                for data in connection.receive_bytes(bytes):
                    yield data
                fd = connection.flush()
                if fd:
                    yield fd
        finally:
            self.close(s)
