import asyncio
import logging
from asyncio.streams import StreamReader, StreamWriter
from collections import deque
from typing import AsyncGenerator, Deque, Dict, Optional

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data

# Define the public API of this module
__all__ = ["AsyncioClient"]


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

    async def write_and_drain(self, data: bytes):
        writer = self.writer
        writer.write(data)
        await writer.drain()

    async def connect(self, host: str, port: int):
        self._reader, self._writer = await asyncio.open_connection(host, port)

    async def close(self):
        writer = self.writer
        self._reader = None
        self._writer = None
        writer.close()
        await writer.wait_closed()


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
        await self._ctrl_stream.connect(self._host, 8888),

        self._ctrl_task = asyncio.create_task(
            self._ctrl_read_forever(self._ctrl_stream.reader)
        )

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

    async def send(self, command: Command[T]) -> T:
        """Send a command to control port of the PandA, returning its response.

        Args:
            command: The `Command` to send
        """
        queue: asyncio.Queue[T] = asyncio.Queue()
        # Need to use the id as non-frozen dataclasses don't hash
        self._ctrl_queues[id(command)] = queue
        to_send = self._ctrl_connection.send(command)
        await self._ctrl_stream.write_and_drain(to_send)
        response = await queue.get()
        if isinstance(response, Exception):
            raise response
        else:
            return response

    async def data(
        self,
        scaled: bool = True,
        flush_period: Optional[float] = None,
        frame_timeout: Optional[float] = None,
    ) -> AsyncGenerator[Data, None]:
        """Connect to data port and yield data frames

        Args:
            scaled: Whether to scale and average data frames, reduces throughput
            flush_period: How often to flush partial data frames, None is on every
                chunk of data from the server
            frame_timeout: If no data is received for this amount of time, raise
                `asyncio.TimeoutError`
        """

        data_stream = _StreamHelper()
        await data_stream.connect(self._host, 8889),

        connection = DataConnection()
        data: Deque[Data] = deque()
        reader = data_stream.reader
        # Should we flush every FrameData?
        flush_every_frame = flush_period is None

        async def queue_flushed_data():
            data.extend(connection.flush())

        async def periodic_flush():
            if not flush_every_frame:
                while True:
                    # Every flush_period seconds flush and queue data
                    await asyncio.gather(
                        asyncio.sleep(flush_period), queue_flushed_data()
                    )

        flush_task = asyncio.create_task(periodic_flush())
        try:
            await data_stream.write_and_drain(connection.connect(scaled))
            while True:
                received = await asyncio.wait_for(reader.read(4096), frame_timeout)
                for d in connection.receive_bytes(received, flush_every_frame):
                    data.append(d)
                while data:
                    yield data.popleft()
        finally:
            flush_task.cancel()
            await data_stream.close()
