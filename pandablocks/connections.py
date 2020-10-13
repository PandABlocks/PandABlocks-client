import struct
import sys
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any, Callable, Deque, Iterator, List, Tuple

import numpy as np

from .commands import Command
from .responses import Data, DataField, EndData, EndReason, FrameData, StartData

# The name of the samples field used for averaging unscaled fields
SAMPLES_FIELD = "PCAP.SAMPLES.Value"


class Buffer:
    """Byte storage that provides line reader and bytes reader interfaces.
    For example::

        buf = Buffer()
        buf += bytes_from_server
        line = buf.read_line()  # raises StopIteration if no line
        for line in buf:
            pass
        bytes = buf.read_bytes(50)  # raises StopIteration if not enough bytes
    """

    def __init__(self):
        self._buf = bytearray()

    def __iadd__(self, byteslike: bytes):
        """Add some data from the server"""
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
        """Read and pop num bytes from the beginning of the buffer"""
        if num > len(self._buf):
            raise StopIteration()
        else:
            return self._extract_frame(num)

    def peek_bytes(self, num: int) -> bytes:
        """Read but do not pop num bytes from the beginning of the buffer"""
        if num > len(self._buf):
            raise StopIteration()
        else:
            return self._buf[:num]

    def read_line(self):
        """Read and pop a newline terminated line (without terminator)
        from the beginning of the buffer"""
        idx = self._buf.find(b"\n")
        if idx < 0:
            raise StopIteration()
        else:
            return self._extract_frame(idx, num_to_discard=1)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return self.read_line()


class ControlConnection:
    """Sans-IO connection to control port of PandA TCP server, supporting a
    Command based interface. For example::

        cc = ControlConnection()
        # Connection says what bytes should be sent to execute command
        to_send = cc.send(command)
        socket.sendall(to_send)
        while True:
            # Repeatedly process bytes from the server looking for responses
            from_server = socket.recv()
            for command, response in cc.receive_bytes(from_server):
                do_something_with(response)
    """

    def __init__(self):
        self._buf = Buffer()
        self._lines: List[bytes] = []
        self._commands: Deque[Command] = deque()

    def receive_bytes(self, received: bytes) -> Iterator[Tuple[Command, Any]]:
        """Tell the connection that you have received some bytes off the network.
        Parse these into high level responses which are yielded back with the
        command that triggered this response"""
        self._buf += received
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


class DataConnection:
    """Sans-IO connection to data port of PandA TCP server, supporting an
    flushable iterator interface. For example::

        dc = ControlConnection()
        # Single connection string to send
        to_send = dc.connect()
        socket.sendall(to_send)
        while True:
            # Repeatedly process bytes from the server looking for data
            from_server = socket.recv()
            for data in dc.receive_bytes(from_server):
                do_something_with(data)
            # FrameData are squashed together until the end of acquisition
            # or an explicit flush is called
            if time_to_flush():
                for data in dc.flush():
                    do_something_with(data)
    """

    def __init__(self):
        # TODO: could support big endian, but are there any systems out there?
        assert sys.byteorder == "little", "PandA sends data little endian"
        self._buf = Buffer()
        self._header = ""
        self._next_handler: Callable = None
        self._frame_dtype = None
        self._scaled_dtype = None
        self._partial_data = bytearray()

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
            data = root.find("data")
            sample_bytes = int(data.get("sample_bytes"))
            if sample_bytes - sum(f.type.itemsize for f in fields) == 4:
                # In raw mode with panda-server < 2.1 samples wasn't
                # put in if not specifically requested, but was still
                # sent
                name, capture = SAMPLES_FIELD.rsplit(".", maxsplit=1)
                fields.insert(
                    0, DataField(name, np.dtype("uint32"), capture),
                )
            self._frame_dtype = np.dtype(
                [(f"{f.name}.{f.capture}", f.type) for f in fields]
            )
            return StartData(
                fields=fields,
                missed=int(data.get("missed")),
                process=str(data.get("process")),
                format=str(data.get("format")),
                sample_bytes=sample_bytes,
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
            for fd in self.flush():
                return fd
        else:
            raise ValueError(f"Bad data '{bytes}'")

    def _handle_data_frame(self):
        # Peek message length as uint32 LE
        length = struct.unpack("<I", self._buf.peek_bytes(4))[0]
        # length = len("BIN " + 4_bytes_encoded_length + data)
        # we already read "BIN ", so read the rest
        self._partial_data += self._buf.read_bytes(length - 4)[4:]
        self._next_handler = self._handle_data_start

    def _handle_data_end(self):
        samples, reason = self._buf.read_line().split(maxsplit=1)
        self._next_handler = self._handle_header_start
        return EndData(samples=int(samples), reason=EndReason(reason.decode()))

    def receive_bytes(self, received: bytes) -> Iterator[Data]:
        """Tell the connection that you have received some bytes off the network.
        Parse these into Data structures and yield them back. Do not yield
        `FrameData` unless `flush` is called or end of acquisition reached"""
        assert self._next_handler, "Connect not called"
        self._buf += received
        while True:
            # Each of these handlers should call read at most once, so that
            # if we don't have enough data we don't lose partial data
            try:
                ret = self._next_handler()
            except StopIteration:
                break
            else:
                if ret:
                    yield ret

    def flush(self) -> Iterator[FrameData]:
        """If there is a partial data frame, pop and yield it"""
        # Some partial data left to return
        if self._partial_data:
            data = np.frombuffer(self._partial_data, self._frame_dtype)
            self._partial_data = bytearray()
            yield FrameData(data)

    def connect(self, scaled: bool) -> bytes:
        """Return the bytes that need to be sent on connection"""
        assert not self._next_handler, "Can't connect while midway through collection"
        self._next_handler = self._handle_header_start
        if scaled:
            return b"XML FRAMED SCALED\n"
        else:
            return b"XML FRAMED RAW\n"
