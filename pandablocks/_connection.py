from typing import Iterator, ByteString, Optional, Sequence

from ._commands import Command
from ._events import Event


class LineBuffer:
    def __init__(self):
        self._data = bytearray()
        self._lines = []

    def __iadd__(self, byteslike: ByteString):
        self._data += byteslike
        return self

    def get_lines(self) -> Optional[Sequence[ByteString]]:
        """Get lines from the buffer, stripping the newline"""
        is_multiline = bool(self._lines)
        while True:
            # Find a line ending
            offset = self._data.find(b"\n")
            if offset == -1:
                # No new lines, return nothing
                return None
            # Consume the line from the data
            line = self._data[:offset]
            self._data = self._data[offset + 1:]
            # . means the end of a multiline transaction
            if line == b".":
                # End of multiline mode
                lines = self._lines
                self._lines = []
                return lines
            # Check if we need to switch to multiline mode
            if not is_multiline:
                is_multiline = line.startswith(b"!") or line == b"."
            if is_multiline:
                # Add a new line to the buffer
                assert line.startswith(b"!"), \
                    "Multiline response %r doesn't start with !" % line
                self._lines.append(line[1:])
            else:
                # Single line mode
                assert not self._lines, \
                    "Multiline response %s not terminated" % self._lines
                return [line]


class Connection:
    def __init__(self):
        self._received = LineBuffer()
        self._to_send = bytearray()

    def receive_data(self, data: ByteString) -> Iterator[Event]:
        """Process some data received by the socket into client Events"""
        self._received += data
        lines = self._received.get_lines()
        while lines is not None:
            yield Event()
            lines = self._received.get_lines()

    def get(self, field: str):
        """Get the value of a field, or star command.
        E.g.
            Get("PCAP.ACTIVE") -> Value("1")
            Get("*IDN") -> Value("PandA 1.1...")
        """
        self._to_send += b"%s\n" % field

    def data_to_send(self) -> ByteString:
        """Work out what bytes to send to the server as a result of commands"""
        out = self._to_send
        self._to_send = bytearray()
        return out


class GetBlocksData(Command):
    """Get the descriptions and field lists of the requested Blocks
    E.g.
        GetBlocksData() -> BlocksData()
    """


class GetPcapBitsLabels(Command):
    """Get the labels for the bit fields in PCAP
    E.g.
        GetPcapBitsLabels() -> PcapBitsLabels()
    """


class GetChanges(Command):
    """Get the changes since the last time this was called
    E.g.
        GetChanges() -> Changes()
    """


@dataclass
class Set(Command):
    """Set the value of a field.
    E.g.
        Set("PCAP.TRIG", "PULSE1.OUT")
    """
    field: str
    value: str
