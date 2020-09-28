import socket
from typing import Iterator

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

    def send(self, command: Command[T]) -> T:
        bytes = self._ctrl_connection.send(command)
        self._ctrl_socket.sendall(bytes)
        bytes = self._ctrl_socket.recv(4096)
        responses = list(self._ctrl_connection.receive_data(bytes))
        assert len(responses) == 1, f"Expected one response, got {responses}"
        c, response = responses[0]
        assert c == command, f"Got command {c}"
        return response

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
