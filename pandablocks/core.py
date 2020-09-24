import struct
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Generic, Iterator, List, Tuple, TypeVar, Union

import numpy as np


class Buffer:
    """Provides line reader and bytes reader interfaces:

        line = buf.read_line()  # raises StopIteration if no line
        for line in buf:
            pass
        bytes = buf.read_bytes(50)  # raises StopIteration if not enough bytes
    """

    def __init__(self):
        self._buf = bytearray()

    def __iadd__(self, byteslike: bytes):
        """Add some data from the server"""
        assert byteslike, "Stream closed"
        self._buf += byteslike
        return self

    def _extract_frame(self, num_to_extract, num_to_discard=0) -> bytes:
        # extract num_to_extract bytes from the start of the buffer
        frame = self._buf[:num_to_extract]
        # Update the buffer in place, to take advantage of bytearray's
        # optimized delete-from-beginning feature.
        del self._buf[: num_to_extract + num_to_discard]
        return frame

    def read_bytes(self, num: int) -> bytes:
        if num > len(self._buf):
            raise StopIteration
        else:
            return self._extract_frame(num)

    def peek_bytes(self, num: int) -> bytes:
        if num > len(self._buf):
            raise StopIteration()
        else:
            return self._buf[:num]

    def read_line(self):
        idx = self._buf.find(b"\n")
        if idx < 0:
            raise StopIteration()
        else:
            return self._extract_frame(idx, num_to_discard=1)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return self.read_line()


T = TypeVar("T")
# One or more lines to send
Lines = Union[bytes, List[bytes]]


class Command(Generic[T]):
    def lines(self) -> Lines:
        """Return lines that should be sent, with no newlines in them"""
        raise NotImplementedError(self)

    def response(self, lines: Lines) -> T:
        """Create a response from the lines received from the PandA"""
        raise NotImplementedError(self)

    def ok_if(self, ok, lines: Lines):
        """If not ok then raise a suitable error message"""
        if not ok:
            raise ValueError("Bad response to command {self}: '{lines}'")


class ControlConnection:
    """An Event based interface like h11"""

    def __init__(self):
        self._buf = Buffer()
        self._lines: List[bytes] = []
        self._commands: Deque[Command] = deque()

    def receive_data(self, data: bytes) -> Iterator[Tuple[Command, Any]]:
        """Tell the connection that you have received some bytes off the network,
        which are parsed into high level responses and yielded back with the
        command that triggered this response"""
        self._buf += data
        is_multiline = bool(self._lines)
        for line in self._buf:
            if not is_multiline:
                # Check if we need to switch to multiline mode
                is_multiline = line.startswith(b"!") or line == b"."
            if line == b".":
                # End of multiline mode, return what we've got
                command = self._commands.popleft()
                yield command, command.response(self._lines)
                self._lines = []
                is_multiline = False
            elif is_multiline:
                # Add a new line to the buffer
                assert line.startswith(b"!"), (
                    "Multiline response %r doesn't start with !" % line
                )
                self._lines.append(line[1:])
            else:
                # Single line mode
                assert not self._lines, (
                    "Multiline response %s not terminated" % self._lines
                )
                command = self._commands.popleft()
                yield command, command.response(line)

    def send(self, command: Command) -> bytes:
        """Tell the connection you want to send an event, and it will return
        some bytes to send down the network
        """
        self._commands.append(command)
        lines = command.lines()
        if isinstance(lines, list):
            lines = b"\n".join(lines)
        return lines + b"\n"


@dataclass
class DataField:
    name: str
    type: np.dtype
    capture: str
    scale: float
    offset: float
    units: str


@dataclass
class StartData:
    fields: List[DataField]
    missed: int
    process: str
    format: str
    sample_bytes: int


@dataclass
class FrameData:
    data: np.ndarray


class EndReason(Enum):
    # Experiment completed without intervention.
    OK = "Ok"
    # Experiment manually completed by *PCAP.DISARM= command.
    DISARMED = "Disarmed"
    # Client disconnect detected.
    EARLY_DISCONNECT = "Early disconnect"
    # Client not taking data quickly or network congestion, internal buffer overflow.
    DATA_OVERRUN = "Data overrun"
    # Triggers too fast for configured data capture.
    FRAMING_ERROR = "Framing error"
    # Probable CPU overload on PandA, should not occur.
    DRIVER_DATA_OVERRUN = "Driver data overrun"
    # Data capture too fast for memory bandwidth.
    DMA_DATA_ERROR = "DMA data error"


@dataclass
class EndData:
    samples: int
    reason: EndReason


class DataConnection:
    def __init__(self):
        self._buf = Buffer()
        self._header = ""
        self._next_handler: Callable = None
        self._rec_dtype = None

    def _handle_header_start(self):
        line = self._buf.read_line()
        if line == b"<header>":
            self._header = line
            self._next_handler = self._handle_header_body

    def _handle_header_body(self):
        line = self._buf.read_line()
        self._header += line
        if line == b"</header>":
            self._next_handler = self._handle_data_start
            fields = []
            root = ET.fromstring(self._header)
            for field in root.find("fields"):
                fields.append(
                    DataField(
                        name=str(field.get("name")),
                        type=np.dtype(field.get("type")),
                        capture=str(field.get("capture")),
                        scale=float(field.get("scale", 1)),
                        offset=float(field.get("offset", 0)),
                        units=str(field.get("units", "")),
                    )
                )
            self._rec_dtype = np.dtype(
                [(f"{f.name}.{f.capture}", f.type) for f in fields]
            )
            data = root.find("data")
            return StartData(
                fields=fields,
                missed=int(data.get("missed")),
                process=str(data.get("process")),
                format=str(data.get("format")),
                sample_bytes=int(data.get("sample_bytes")),
            )

    def _handle_data_start(self):
        # We might have some leading newlines to string
        newlines = 0
        while self._buf.peek_bytes(newlines + 1) == b"\n" * (newlines + 1):
            newlines += 1
        bytes = self._buf.read_bytes(newlines + 4)[newlines:]
        if bytes == b"BIN ":
            self._next_handler = self._handle_data_frame
        elif bytes == b"END ":
            self._next_handler = self._handle_data_end
        else:
            raise ValueError(f"Bad data '{bytes}'")

    def _handle_data_frame(self):
        # Peek message length as uint32 LE
        length = struct.unpack("<I", self._buf.peek_bytes(4))[0]
        # length = len("BIN " + 4_bytes_encoded_length + data)
        # we already read "BIN ", so read the rest
        packet = self._buf.read_bytes(length - 4)[4:]
        self._next_handler = self._handle_data_start
        return FrameData(np.frombuffer(packet, self._rec_dtype))

    def _handle_data_end(self):
        samples, reason = self._buf.read_line().split()
        self._next_handler = self._handle_header_start
        return EndData(samples=int(samples), reason=EndReason(reason.decode()))

    def receive_data(
        self, data: bytes
    ) -> Iterator[Union[StartData, FrameData, EndData]]:
        """Tell the connection that you have received some bytes off the network,
        which are parsed into high level data structures yielded back"""
        assert self._next_handler, "Connect not called"
        self._buf += data
        while True:
            # Each of these handlers should call read at most once, so that
            # if we don't have enough data we don't lose partial data
            try:
                ret = self._next_handler()
            except StopIteration:
                return
            else:
                if ret:
                    yield ret

    def connect(self) -> bytes:
        """Return the bytes that need to be sent on connection"""
        assert not self._next_handler, "Can't connect while midway through collection"
        self._next_handler = self._handle_header_start
        return b"XML FRAMED SCALED\n"
