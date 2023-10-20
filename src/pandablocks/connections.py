import struct
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterator, List, Optional, Tuple

import numpy as np

from ._exchange import Exchange, ExchangeGenerator, Exchanges
from .commands import Command, CommandException
from .responses import (
    Data,
    EndData,
    EndReason,
    FieldCapture,
    FrameData,
    ReadyData,
    StartData,
)

# Define the public API of this module
__all__ = [
    "NeedMoreData",
    "NoContextAvailable",
    "Buffer",
    "ControlConnection",
    "DataConnection",
]

# The name of the samples field used for averaging unscaled fields
SAMPLES_FIELD = "PCAP.SAMPLES.Value"


class NeedMoreData(Exception):
    """Raised if the `Buffer` isn't full enough to return the requested bytes"""


class NoContextAvailable(Exception):
    """Raised if there were no contexts available for this connection.
    This may result from calling `ControlConnection.receive_bytes()` without calling
    `ControlConnection.send()`, or if there were unmatched sends/receives"""


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


@dataclass
class _ExchangeContext:
    #: The exchange we should be filling
    exchange: Exchange
    #: The command that produced it
    command: Command
    #: If this was the last in the list, the generator to call next
    generator: Optional[ExchangeGenerator[Any]] = None

    def exception(self, e: Exception) -> CommandException:
        """Return a `CommandException` with the sent and received strings
        in the text"""
        msg = f"{self.command} ->"
        if self.exchange.is_multiline:
            for line in self.exchange.multiline:
                msg += "\n    " + line
        else:
            msg += " " + self.exchange.line
        if e.args:
            msg += f"\n{type(e).__name__}:{e}"
        return CommandException(msg).with_traceback(e.__traceback__)


class ControlConnection:
    """Sans-IO connection to control port of PandA TCP server, supporting a
    Command based interface. For example::

        cc = ControlConnection()
        # Connection says what bytes should be sent to execute command
        to_send = cc.send(command)
        socket.sendall(to_send)
        while True:
            # Repeatedly process bytes from the PandA
            received = socket.recv()
            # Sending any subsequent bytes to be sent back to the PandA
            to_send = cc.receive_bytes(received)
            socket.sendall(to_send)
            # And processing the produced responses
            for command, response in cc.responses()
                do_something_with(response)
    """

    def __init__(self) -> None:
        self._buf = Buffer()
        self._lines: List[str] = []
        self._contexts: Deque[_ExchangeContext] = deque()
        self._responses: Deque[Tuple[Command, Any]] = deque()

    def _update_contexts(self, lines: List[str], is_multiline=False) -> bytes:
        to_send = b""
        if len(self._contexts) == 0:
            raise NoContextAvailable()
        context = self._contexts.popleft()
        # Update the exchange with what we've got
        context.exchange.received = lines
        context.exchange.is_multiline = is_multiline
        # If we're given a generator to run then do so
        if context.generator:
            try:
                # Return the bytes from sending the next bit of the command
                exchanges = next(context.generator)
            except StopIteration as e:
                # Command complete, store the result
                self._responses.append((context.command, e.value))
            except Exception as e:
                # Command failed, store an exception
                self._responses.append((context.command, context.exception(e)))
            else:
                to_send = b"".join(
                    self._bytes_from_exchanges(
                        exchanges, context.command, context.generator
                    )
                )
        return to_send

    def _bytes_from_exchanges(
        self, exchanges: Exchanges, command: Command, generator: ExchangeGenerator[Any]
    ) -> Iterator[bytes]:
        if not isinstance(exchanges, list):
            exchanges = [exchanges]
        # No Exchanges when a Command's yield is empty e.g. unexpected/unparseable data
        # received from PandA
        if len(exchanges) == 0:
            return
        for ex in exchanges:
            context = _ExchangeContext(ex, command)
            self._contexts.append(context)
            text = "\n".join(ex.to_send) + "\n"
            yield text.encode()
        # The last exchange gets the generator so it triggers the next thing to send
        context.generator = generator

    def receive_bytes(self, received: bytes) -> bytes:
        """Tell the connection that you have received some bytes off the network.
        Parse these into high level responses which are yielded back by `responses`.
        Return any bytes to send back"""
        self._buf += received
        is_multiline = bool(self._lines)
        to_send = b""
        for line_b in self._buf:
            line = line_b.decode()
            if not is_multiline:
                # Check if we need to switch to multiline mode
                is_multiline = line.startswith("!") or line == "."
            if is_multiline:
                # Add a new line to the buffer
                self._lines.append(line)
                if line == ".":
                    # End of multiline mode, return what we've got
                    to_send += self._update_contexts(self._lines, is_multiline)
                    self._lines = []
                    is_multiline = False
                else:
                    # Check a correctly formatted response
                    assert line.startswith("!"), (
                        "Multiline response %r doesn't start with !" % line
                    )
            else:
                # Single line mode
                assert not self._lines, (
                    "Multiline response %s not terminated" % self._lines
                )
                to_send += self._update_contexts([line])
        return to_send

    def responses(self) -> Iterator[Tuple[Command, Any]]:
        """Get the (command, response) tuples generated as part of the last
        receive_bytes"""
        while self._responses:
            yield self._responses.popleft()

    def send(self, command: Command) -> bytes:
        """Tell the connection you want to send an event, and it will return
        some bytes to send down the network
        """
        # If not given a partially run generator, start one here
        generator = command.execute()
        exchanges = next(generator)
        to_send = b"".join(self._bytes_from_exchanges(exchanges, command, generator))
        return to_send


class DataConnection:
    """Sans-IO connection to data port of PandA TCP server, supporting an
    flushable iterator interface. For example::

        dc = DataConnection()
        # Single connection string to send
        to_send = dc.connect()
        socket.sendall(to_send)
        while True:
            # Repeatedly process bytes from the PandA looking for data
            received = socket.recv()
            for data in dc.receive_bytes(received):
                do_something_with(data)
    """

    def __init__(self) -> None:
        # TODO: could support big endian, but are there any systems out there?
        assert sys.byteorder == "little", "PandA sends data little endian"
        # Store of bytes received so far to parse in the handlers
        self._buf = Buffer()
        # Header text from PandA with field info
        self._header = ""
        # The next parsing handler that should be called if there is data in buffer
        self._next_handler: Optional[Callable[[], Optional[Iterator[Data]]]] = None
        # numpy dtype of produced FrameData
        self._frame_dtype = None
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
                    0,
                    FieldCapture(name, np.dtype("uint32"), capture),
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
        self._partial_data += data
        # if told to flush now, then yield what we have
        if self._flush_every_frame:
            yield from self.flush()
        self._next_handler = self._handle_data_start

    def _handle_data_end(self):
        # Handle the end reason
        samples, reason = self._buf.read_line().split(maxsplit=1)
        reason_enum = EndReason(reason.decode())
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
