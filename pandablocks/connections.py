import struct
import sys
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any, Callable, Deque, Iterator, List, Optional, Tuple

import numpy as np

from .commands import Command, Lines
from .responses import (
    Data,
    EndData,
    EndReason,
    FieldCapture,
    FrameData,
    ReadyData,
    StartData,
)

# The name of the samples field used for averaging unscaled fields
SAMPLES_FIELD = "PCAP.SAMPLES.Value"


class NeedMoreData(Exception):
    """Raised if the `Buffer` isn't full enough to return the requested bytes"""


class Buffer:
    """Byte storage that provides line reader and bytes reader interfaces.
    For example::

        buf = Buffer()
        buf += bytes_from_server
        line = buf.read_line()  # raises NeedMoreData if no line
        for line in buf:
            pass
        bytes = buf.read_bytes(50)  # raises NeedMoreData if not enough bytes
    """

    def __init__(self):
        self._buf = bytearray()

    def __iadd__(self, byteslike: bytes):
        """Add some data from the server"""
        self._buf += byteslike
        return self

    def _extract_frame(self, num_to_extract, num_to_discard=0) -> bytearray:
        # extract num_to_extract bytes from the start of the buffer
        frame = self._buf[:num_to_extract]
        # Update the buffer in place, to take advantage of bytearray's
        # optimized delete-from-beginning feature.
        del self._buf[: num_to_extract + num_to_discard]
        return frame

    def read_bytes(self, num: int) -> bytearray:
        """Read and pop num bytes from the beginning of the buffer, raising
        `NeedMoreData` if the buffer isn't full enough to do so"""
        if num > len(self._buf):
            raise NeedMoreData()
        else:
            return self._extract_frame(num)

    def peek_bytes(self, num: int) -> bytearray:
        """Read but do not pop num bytes from the beginning of the buffer,
        raising `NeedMoreData` if the buffer isn't full enough to do so"""
        if num > len(self._buf):
            raise NeedMoreData()
        else:
            return self._buf[:num]

    def read_line(self):
        """Read and pop a newline terminated line (without terminator)
        from the beginning of the buffer, raising `NeedMoreData` if the
        buffer isn't full enough to do so"""
        idx = self._buf.find(b"\n")
        if idx < 0:
            raise NeedMoreData()
        else:
            return self._extract_frame(idx, num_to_discard=1)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        try:
            return self.read_line()
        except NeedMoreData:
            raise StopIteration()


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

    def _pop_command_response(self, lines: Lines) -> Tuple[Command, Any]:
        command = self._commands.popleft()
        try:
            response = command.response(lines)
        except Exception as e:
            response = e
        return command, response

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
                yield self._pop_command_response(self._lines)
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
                yield self._pop_command_response(line)

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
    """

    def __init__(self):
        # TODO: could support big endian, but are there any systems out there?
        assert sys.byteorder == "little", "PandA sends data little endian"
        # Store of bytes received so far to parse in the handlers
        self._buf = Buffer()
        # Header text from PandA with field info
        self._header = ""
        # The next parsing handler that should be called if there is data in buffer
        self._next_handler: Callable[[], Optional[Iterator[Data]]] = None
        # numpy dtype of produced FrameData
        self._frame_dtype = None
        # frame data that has been received but should be discarded on Data Overrun
        self._pending_data = bytearray()
        # frame data that has been received but not flushed yet
        self._partial_data = bytearray()
        # whether to flush after every frame
        self._flush_every_frame = False

    def _handle_connected(self):
        # Get the response from connect()
        line = self._buf.read_line()
        assert line == b"OK", f"Expected OK, got {line!r}"
        yield ReadyData()
        self._next_handler = self._handle_header_start

    def _handle_header_start(self):
        # Discard lines until we see header start tag
        line = self._buf.read_line()
        if line == b"<header>":
            self._header = line
            self._next_handler = self._handle_header_body

    def _handle_header_body(self):
        # Accumumlate header until the end tag, then parese and return
        line = self._buf.read_line()
        self._header += line
        if line == b"</header>":
            fields = []
            root = ET.fromstring(self._header)
            for field in root.find("fields"):
                fields.append(
                    FieldCapture(
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
                    0, FieldCapture(name, np.dtype("uint32"), capture),
                )
            self._frame_dtype = np.dtype(
                [(f"{f.name}.{f.capture}", f.type) for f in fields]
            )
            yield StartData(
                fields=fields,
                missed=int(data.get("missed")),
                process=str(data.get("process")),
                format=str(data.get("format")),
                sample_bytes=sample_bytes,
            )
            self._next_handler = self._handle_header_end

    def _handle_header_end(self):
        # Discard the newline at the end of the header
        assert self._buf.read_bytes(1) == b"\n", "Expected newline at end of header"
        self._next_handler = self._handle_data_start

    def _handle_data_start(self):
        # Handle "BIN " or "END "
        bytes = self._buf.read_bytes(4)
        if bytes == b"BIN ":
            self._next_handler = self._handle_data_frame
        elif bytes == b"END ":
            self._next_handler = self._handle_data_end
        else:
            raise ValueError(f"Bad data '{bytes}'")

    def _handle_data_frame(self):
        # Handle a whole data frame
        # Peek message length as uint32 LE
        # length = len("BIN " + 4_bytes_encoded_length + data)
        length = struct.unpack("<I", self._buf.peek_bytes(4))[0]
        # we already read "BIN ", so read the rest
        data = self._buf.read_bytes(length - 4)[4:]
        # The pending data is now valid, and what we got is pending
        self._partial_data += self._pending_data
        self._pending_data = data
        # if told to flush now, then yield what we have
        if self._flush_every_frame:
            yield from self.flush()
        self._next_handler = self._handle_data_start

    def _handle_data_end(self):
        # Handle the end reason
        samples, reason = self._buf.read_line().split(maxsplit=1)
        reason_enum = EndReason(reason.decode())
        if reason_enum != EndReason.DATA_OVERRUN:
            # The last bit of pending data is now valid
            self._partial_data += self._pending_data
        # Flush whatever is not already flushed
        yield from self.flush()
        yield EndData(samples=int(samples), reason=reason_enum)
        self._next_handler = self._handle_header_start

    def receive_bytes(self, received: bytes, flush_every_frame=True) -> Iterator[Data]:
        """Tell the connection that you have received some bytes off the network.
        Parse these into Data structures and yield them back.

        Args:
            received: the bytes you received from the socket
            flush_every_frame: Whether to flush `FrameData` as soon as received.
                If False then they will only be sent if `flush` is called or
                end of acquisition reached
        """
        assert self._next_handler, "Connect not called"
        self._flush_every_frame = flush_every_frame
        self._buf += received
        while True:
            # Each of these handlers should call read at most once, so that
            # if we don't have enough data we don't lose partial data
            try:
                ret = self._next_handler()
                if ret:
                    # This is an iterator of Data objects
                    yield from ret
            except NeedMoreData:
                break

    def flush(self) -> Iterator[FrameData]:
        """If there is a partial data frame, pop and yield it"""
        if self._partial_data:
            # Make a numpy array wrapper to the bytearray, no copying here
            data = np.frombuffer(self._partial_data, self._frame_dtype)
            # Make a new bytearray, numpy view will keep the reference
            # to the old one so can't clear it in place
            self._partial_data = bytearray()
            yield FrameData(data)

    def connect(self, scaled: bool) -> bytes:
        """Return the bytes that need to be sent on connection"""
        assert not self._next_handler, "Can't connect while midway through collection"
        self._next_handler = self._handle_connected
        if scaled:
            return b"XML FRAMED SCALED\n"
        else:
            return b"XML FRAMED RAW\n"
