import asyncio
import logging
from asyncio.streams import StreamReader, StreamWriter
from contextlib import suppress
from enum import Enum
from typing import AsyncGenerator, Dict, Iterable, Optional

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data

# Define the public API of this module
__all__ = ["AsyncioClient", "FlushMode"]


class _StreamHelper:
    _reader: Optional[StreamReader] = None
    _writer: Optional[StreamWriter] = None

    @property
    def reader(self) -> StreamReader:
        assert self._reader, "connect() not called yet"
        return self._reader

    @property
    def writer(self) -> StreamWriter:
        assert self._writer, "connect() not called yet"
        return self._writer

    async def write_and_drain(self, data: bytes, timeout: Optional[float] = None):
        writer = self.writer
        writer.write(data)

        # Cannot simply await the drain, as if the remote end has disconnected
        # then the drain will never complete as the OS cannot clear its send buffer.
        write_task = asyncio.create_task(writer.drain())
        _, pending = await asyncio.wait([write_task], timeout=timeout)
        if len(pending):
            for task in pending:
                task.cancel()
            raise asyncio.TimeoutError("Timeout writing data")

    async def connect(self, host: str, port: int):
        self._reader, self._writer = await asyncio.open_connection(host, port)

    async def close(self):
        writer = self.writer
        self._reader = None
        self._writer = None
        writer.close()
        await writer.wait_closed()


class FlushMode(Enum):
    """
    The mode which `AsyncioClient.data()` uses when flushing data frames.
    """

    #: Flush all data frames immediately.
    IMMEDIATE = 0

    #: Flush data frames periodically.
    PERIODIC = 1

    #: Flush data frames when the user sets an `asyncio.Event`.
    MANUAL = 2


class AsyncioClient:
    """Asyncio implementation of a PandABlocks client.
    For example::

        async with AsyncioClient("hostname-or-ip") as client:
            # Control port is now connected
            resp1, resp2 = await asyncio.gather(client.send(cmd1), client.send(cmd2))
            resp3 = await client.send(cmd3)
            async for data in client.data():
                handle(data)
        # Control and data ports are now disconnected
    """

    def __init__(self, host: str):
        self._host = host
        self._ctrl_connection = ControlConnection()
        self._ctrl_task: Optional[asyncio.Task] = None
        self._ctrl_queues: Dict[int, asyncio.Queue] = {}
        self._ctrl_stream = _StreamHelper()

    async def connect(self):
        """Connect to the control port, and be ready to handle commands"""
        await self._ctrl_stream.connect(self._host, 8888)

        self._ctrl_task = asyncio.create_task(
            self._ctrl_read_forever(self._ctrl_stream.reader)
        )

    def is_connected(self):
        """True if there is a currently active connection.
        NOTE: This does not indicate if the remote end is still connected."""
        if self._ctrl_task and not self._ctrl_task.done():
            return True
        return False

    async def close(self):
        """Close the control connection, and wait for completion"""
        assert self._ctrl_task, "connect() not called yet"
        self._ctrl_task.cancel()
        await self._ctrl_stream.close()

    async def __aenter__(self) -> "AsyncioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ctrl_read_forever(self, reader: asyncio.StreamReader):
        while True:
            received = await reader.read(4096)
            try:
                to_send = self._ctrl_connection.receive_bytes(received)
                await self._ctrl_stream.write_and_drain(to_send)
                for command, response in self._ctrl_connection.responses():
                    queue = self._ctrl_queues.pop(id(command))
                    queue.put_nowait(response)
            except Exception:
                logging.exception(f"Error handling '{received.decode()}'")

    async def send(self, command: Command[T], timeout: Optional[float] = None) -> T:
        """Send a command to control port of the PandA, returning its response.

        Args:
            command: The `Command` to send
        """
        queue: asyncio.Queue[T] = asyncio.Queue()
        # Need to use the id as non-frozen dataclasses don't hash
        self._ctrl_queues[id(command)] = queue
        to_send = self._ctrl_connection.send(command)
        await self._ctrl_stream.write_and_drain(to_send, timeout)
        response = await asyncio.wait_for(queue.get(), timeout)
        if isinstance(response, Exception):
            raise response
        else:
            return response

    async def data(
        self,
        scaled: bool = True,
        flush_period: Optional[float] = None,
        frame_timeout: Optional[float] = None,
        flush_event: Optional[asyncio.Event] = None,
        flush_mode: FlushMode = FlushMode.IMMEDIATE,
    ) -> AsyncGenerator[Data, None]:
        """Connect to data port and yield data frames.

        Args:
            scaled: Whether to scale and average data frames, reduces throughput
            flush_period: How often to flush partial data frames when ``flush_mode``
                is ``PERIODIC``.
            frame_timeout: If no data is received for this amount of time, raise
                `asyncio.TimeoutError`.
            flush_event: An `asyncio.Event` to manually flush. When set while
                ``flush_mode`` is ``MANUAL`` a flush will be performed and the event
                will be unset.
            flush_mode: Which mode of the `FlushMode` values.
        """

        # Check input
        if flush_mode == FlushMode.MANUAL:
            if not flush_event:
                raise ValueError(
                    f"flush_event cannot be {flush_event} if flush_mode is 'MANUAL'"
                )
            if flush_period:
                raise ValueError(
                    f"Unused flush_period={flush_period} "
                    "inputted during `MANUAL` flush mode"
                )
        elif flush_mode == FlushMode.PERIODIC:
            if not flush_period:
                raise ValueError(
                    f"flush_period cannot be {flush_period} "
                    "if flush_mode is 'PERIODIC', use `MANUAL` "
                    "flush_mode to make flushing exclusively "
                    "manual"
                )
        elif flush_mode == FlushMode.IMMEDIATE:
            if flush_event:
                raise ValueError(
                    f"Unused flush_event={flush_event} "
                    "inputted during `IMMEDIATE` flush mode"
                )
            if flush_period:
                raise ValueError(
                    f"Unused flush_period={flush_period} "
                    "inputted during `IMMEDIATE` flush mode"
                )
        else:
            raise ValueError(
                "flush_mode must be one of 'IMMEDIATE', 'PERIODIC', 'MANUAL'"
            )

        stream = _StreamHelper()
        connection = DataConnection()
        queue: asyncio.Queue[Iterable[Data]] = asyncio.Queue()
        flush_every_frame = flush_mode == FlushMode.IMMEDIATE
        flush_event = flush_event or asyncio.Event()

        def raise_timeouterror():
            raise asyncio.TimeoutError(f"No data received for {frame_timeout}s")
            yield

        async def flush_loop():
            if flush_every_frame:
                # If flush mode is `IMMEDIATE` flushing will be performed
                # whenever possible
                return
            while True:
                try:
                    await asyncio.wait_for(flush_event.wait(), flush_period)
                except asyncio.TimeoutError:
                    pass
                else:
                    flush_event.clear()
                queue.put_nowait(connection.flush())

        async def read_from_stream():
            reader = stream.reader
            while True:
                try:
                    recv = await asyncio.wait_for(reader.read(4096), frame_timeout)
                except asyncio.TimeoutError:
                    queue.put_nowait(raise_timeouterror())
                    break
                else:
                    queue.put_nowait(
                        connection.receive_bytes(
                            recv, flush_every_frame=flush_every_frame
                        )
                    )

        await stream.connect(self._host, 8889)
        await stream.write_and_drain(connection.connect(scaled))
        fut = asyncio.gather(read_from_stream(), flush_loop())
        try:
            while True:
                for data in await queue.get():
                    yield data
        finally:
            fut.cancel()
            await stream.close()
            with suppress(asyncio.CancelledError):
                await fut
